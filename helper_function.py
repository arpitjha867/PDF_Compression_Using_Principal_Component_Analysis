import json
from PIL import Image
import io
import numpy as np

# compression limits
class CompressionConfig:
    """PCA-based compression settings for different region types"""

    # text regions
    TEXT_COMPONENTS = 80
    TEXT_MIN_VARIANCE = 90.0
    TEXT_JPEG_QUALITY = 85

    # image regions
    IMAGE_COMPONENTS = 80
    IMAGE_MIN_VARIANCE = 95.0
    IMAGE_MAX_SIZE = 800
    IMAGE_JPEG_QUALITY = 85

    # unknown regions
    UNKNOWN_COMPONENTS = 70
    UNKNOWN_MIN_VARIANCE = 92.0
    UNKNOWN_JPEG_QUALITY = 80
    
    # Smart compression thresholds
    MIN_SIZE_FOR_PCA = 50 * 1024  # Only use PCA for images > 50KB
    MIN_DIMENSION_FOR_PCA = 200   # Only use PCA for images > 200x200px


# main pca logic
class PCAImageCompressor:
    """PCA-based image compression using grayscale for performance."""

    def __init__(self):
        self.explained_variance = None

    def compress_channel(self, channel_data, n_components):
        
        height, width = channel_data.shape
        data = channel_data.astype(np.float64)
        mean = np.mean(data, axis=0)
        centered_data = data - mean

        # singular value decomposition svd
        U, S, VT = np.linalg.svd(centered_data, full_matrices=False)
        n_components = min(n_components, min(height, width))

        U_r = U[:, :n_components]
        S_r = S[:n_components]
        VT_r = VT[:n_components, :]

        reconstructed = (U_r @ np.diag(S_r)) @ VT_r + mean

        # Explained variance
        total_var = np.sum(S ** 2)
        explained_var = np.sum(S_r ** 2) / total_var * 100
        self.explained_variance = explained_var

        # Compression ratio
        original_bytes = height * width
        compressed_bytes = U_r.size + S_r.size + VT_r.size + mean.size
        compression_ratio = original_bytes / compressed_bytes

        return reconstructed, compression_ratio, explained_var

    def compress_image(self, image_array, n_components, min_variance=None):
        if len(image_array.shape) == 2:  # grayscale
            channels = [image_array]
        else:  # RGB
            channels = [image_array[:, :, i] for i in range(3)]

        compressed_channels = []
        variances = []
        ratios = []

        for ch in channels:
            current_components = n_components
            if min_variance:
                current_components = self._find_components_for_variance(
                    ch, n_components, min_variance
                )
            comp_ch, ratio, variance = self.compress_channel(ch, current_components)
            compressed_channels.append(np.clip(comp_ch, 0, 255).astype(np.uint8))
            variances.append(variance)
            ratios.append(ratio)

        # Recombine channels
        if len(compressed_channels) == 1:
            compressed = compressed_channels[0]
        else:
            compressed = np.stack(compressed_channels, axis=2)

        return compressed, np.mean(ratios), np.mean(variances)

    def _find_components_for_variance(self, channel_data, initial_components, min_variance):
        _, _, variance = self.compress_channel(channel_data, initial_components)
        if variance >= min_variance:
            return initial_components

        estimated = int(initial_components * (min_variance / variance))
        return min(estimated, min(channel_data.shape))


# === HELPER FUNCTIONS ===
def load_metadata(json_path):
    """Load region metadata from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def simple_jpeg_compress(img, quality, region_type):
    """Simple JPEG compression without PCA."""
    output = io.BytesIO()
    
    # Preserve original mode if grayscale
    save_mode = img.mode if img.mode in ('L', 'RGB') else 'RGB'
    if save_mode != img.mode:
        img = img.convert(save_mode)
    
    img.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()


def compress_image_smart(img_data, region_type, original_dpi=None):
    """
    Compress image using PCA based on region type.
    Falls back to simple JPEG for small images to avoid size increase.
    """
    if isinstance(img_data, bytes):
        img = Image.open(io.BytesIO(img_data))
        original_bytes_size = len(img_data)
    else:
        img = img_data
        # Estimate original size
        output_temp = io.BytesIO()
        img.save(output_temp, format=img.format or "PNG")
        original_bytes_size = len(output_temp.getvalue())

    # Store original mode
    original_mode = img.mode
    
    # Get dimensions
    width, height = img.size
    img_array = np.array(img)
    original_array_size = img_array.nbytes

    # === DECISION LOGIC: PCA vs Simple JPEG ===
    
    # Check 1: Is image too small for PCA to be beneficial?
    use_simple_jpeg = (
        original_bytes_size < CompressionConfig.MIN_SIZE_FOR_PCA or
        min(width, height) < CompressionConfig.MIN_DIMENSION_FOR_PCA
    )
    
    # Get compression parameters based on region type
    if region_type == "text":
        n_components = CompressionConfig.TEXT_COMPONENTS
        min_variance = CompressionConfig.TEXT_MIN_VARIANCE
        jpeg_quality = CompressionConfig.TEXT_JPEG_QUALITY
    elif region_type == "image":
        n_components = CompressionConfig.IMAGE_COMPONENTS
        min_variance = CompressionConfig.IMAGE_MIN_VARIANCE
        jpeg_quality = CompressionConfig.IMAGE_JPEG_QUALITY
    else:
        n_components = CompressionConfig.UNKNOWN_COMPONENTS
        min_variance = CompressionConfig.UNKNOWN_MIN_VARIANCE
        jpeg_quality = CompressionConfig.UNKNOWN_JPEG_QUALITY

    # === SIMPLE JPEG PATH (for small images) ===
    if use_simple_jpeg:
        try:
            compressed_bytes = simple_jpeg_compress(img, jpeg_quality, region_type)
            
            # Only use compressed version if it's actually smaller
            if len(compressed_bytes) >= original_bytes_size:
                # Return original if compression made it bigger
                if isinstance(img_data, bytes):
                    compressed_bytes = img_data
                else:
                    output = io.BytesIO()
                    img.save(output, format=img.format or "PNG")
                    compressed_bytes = output.getvalue()
            
            stats = {
                "original_size": original_bytes_size,
                "compressed_size": len(compressed_bytes),
                "compression_ratio": original_bytes_size / len(compressed_bytes),
                "variance_explained": 0,
                "components_used": 0,
                "region_type": region_type,
                "method": "simple_jpeg" if len(compressed_bytes) < original_bytes_size else "original"
            }
            return compressed_bytes, stats
            
        except Exception as e:
            # Return original on error
            if isinstance(img_data, bytes):
                return img_data, {
                    "original_size": original_bytes_size,
                    "compressed_size": original_bytes_size,
                    "compression_ratio": 1.0,
                    "variance_explained": 0,
                    "components_used": 0,
                    "region_type": region_type,
                    "method": "original",
                    "error": str(e)
                }

    # === PCA PATH (for larger images) ===
    try:
        # Convert to RGB for PCA processing
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        
        img_array = np.array(img)

        # Optional downsampling for very large images
        max_dimension = CompressionConfig.IMAGE_MAX_SIZE
        if max(img.size) > max_dimension:
            scale = max_dimension / max(img.size)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            img_array = np.array(img)

        compressor = PCAImageCompressor()

        # PCA compression
        compressed_array, ratio, variance = compressor.compress_image(
            img_array, n_components=n_components, min_variance=min_variance
        )

        # Convert back to image
        compressed_img = Image.fromarray(compressed_array).convert("RGB")

        # Save as JPEG
        output = io.BytesIO()
        compressed_img.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
        compressed_bytes = output.getvalue()

        compressed_size = len(compressed_bytes)
        
        # Check if PCA actually helped
        if compressed_size >= original_bytes_size * 0.95:  # If not at least 5% savings
            # Fall back to simple JPEG
            compressed_bytes = simple_jpeg_compress(img, jpeg_quality, region_type)
            compressed_size = len(compressed_bytes)
            method = "fallback_jpeg"
            variance = 0
            n_components = 0
        else:
            method = "pca"

        # Final check: if still bigger, return original
        if compressed_size >= original_bytes_size:
            if isinstance(img_data, bytes):
                compressed_bytes = img_data
            else:
                output = io.BytesIO()
                img.save(output, format=img.format or "PNG")
                compressed_bytes = output.getvalue()
            compressed_size = len(compressed_bytes)
            method = "original"

        final_ratio = original_bytes_size / compressed_size if compressed_size > 0 else 1.0

        stats = {
            "original_size": original_bytes_size,
            "compressed_size": compressed_size,
            "compression_ratio": final_ratio,
            "variance_explained": variance,
            "components_used": n_components,
            "region_type": region_type,
            "method": method
        }
        return compressed_bytes, stats

    except Exception as e:
        # Final fallback: simple JPEG or original
        try:
            compressed_bytes = simple_jpeg_compress(img, jpeg_quality, region_type)
            if len(compressed_bytes) >= original_bytes_size:
                if isinstance(img_data, bytes):
                    compressed_bytes = img_data
                else:
                    output = io.BytesIO()
                    img.save(output, format=img.format or "PNG")
                    compressed_bytes = output.getvalue()
            
            stats = {
                "original_size": original_bytes_size,
                "compressed_size": len(compressed_bytes),
                "compression_ratio": original_bytes_size / len(compressed_bytes),
                "variance_explained": 0,
                "components_used": 0,
                "region_type": region_type,
                "method": "error_fallback",
                "error": str(e)
            }
            return compressed_bytes, stats
        except:
            # Absolute last resort: return original
            if isinstance(img_data, bytes):
                return img_data, {
                    "original_size": original_bytes_size,
                    "compressed_size": original_bytes_size,
                    "compression_ratio": 1.0,
                    "variance_explained": 0,
                    "components_used": 0,
                    "region_type": region_type,
                    "method": "original",
                    "error": str(e)
                }