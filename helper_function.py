import json
from PIL import Image
import io
import numpy as np


class CompressionConfig:
    # image regions
    IMAGE_COMPONENTS = 80
    IMAGE_MIN_VARIANCE = 92.0
    IMAGE_MAX_SIZE = 800
    IMAGE_JPEG_QUALITY = 85

    # unknown regions
    UNKNOWN_COMPONENTS = 70
    UNKNOWN_MIN_VARIANCE = 92.0
    UNKNOWN_JPEG_QUALITY = 80

    # Smart compression thresholds
    MIN_SIZE_FOR_PCA = 50 * 1024  # Only use PCA for images > 50KB
    MIN_DIMENSION_FOR_PCA = 200   # Only use PCA for images > 200x200px


class PCAImageCompressor:
    """PCA-based image compression using per-channel SVD."""

    def compress_image(self, image_array, max_components, min_variance=None):
        if image_array.ndim == 2:  # grayscale
            channels = [image_array]
        else:  # RGB
            channels = [image_array[:, :, i] for i in range(3)]
        compressed_channels = []
        ratios = []
        variances = []

        for ch in channels:
            comp_ch, ratio, variance = self.compress_single_channel(
                ch,
                max_components=max_components,
                min_variance=min_variance,
            )
            compressed_channels.append(comp_ch)
            ratios.append(ratio)
            variances.append(variance)

        if len(compressed_channels) == 1:
            compressed = compressed_channels[0]
        else:
            compressed = np.stack(compressed_channels, axis=2)

        return compressed, float(np.mean(ratios)), float(np.mean(variances))

    def compress_single_channel(self, channel_data, max_components, min_variance):
        height, width = channel_data.shape
        data = channel_data.astype(np.float64)
        mean = data.mean(axis=0)
        centered = data - mean

        # svd
        U, S, VT = np.linalg.svd(centered, full_matrices=False)

        # limit by geometry eg cant chose 100 if img is 200*80 -> 80
        max_valid_components = min(max_components, height, width)
        if max_valid_components <= 0 or S.size == 0:
            # Degenerate case if svd gives no singular matrix
            recon = np.clip(data, 0, 255).astype(np.uint8)
            return recon, 1.0, 0.0

        # choose components
        total_var = np.sum(S ** 2)
        if total_var == 0: # image is flat every poixcel is same
            k = 1
            explained_var = 0.0
        else:
            cum_var = np.cumsum(S ** 2) / total_var * 100.0
            # smallest k such that variance >= min_variance, but not above max_valid_components
            k = max_valid_components
            for idx in range(max_valid_components):
                if cum_var[idx] >= min_variance:
                    k = idx + 1
                    break
            explained_var = cum_var[k - 1]

        # reconstruction with k components
        U_r = U[:, :k]
        S_r = S[:k]
        VT_r = VT[:k, :]

        recon = (U_r @ np.diag(S_r)) @ VT_r + mean

        # compression ratio (approximate: number of stored values vs pixels)
        original_bytes = height * width
        compressed_bytes = U_r.size + S_r.size + VT_r.size + mean.size
        if compressed_bytes <= 0:
            compression_ratio = 1.0
        else:
            compression_ratio = original_bytes / compressed_bytes

        recon = np.clip(recon, 0, 255).astype(np.uint8)
        return recon, compression_ratio, explained_var


def load_metadata(json_path):
    """Load region metadata from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def simple_jpeg_compress(img, quality, dpi=None):
    output = io.BytesIO()

    # Preserve original mode if grayscale or RGB
    save_mode = img.mode if img.mode in ("L", "RGB") else "RGB"
    if save_mode != img.mode:
        img = img.convert(save_mode)

    save_kwargs = {
        "format": "JPEG",
        "quality": quality,
        "optimize": True,
    }
    if dpi is not None:
        save_kwargs["dpi"] = (int(dpi), int(dpi))

    img.save(output, **save_kwargs)
    return output.getvalue()


def compress_image_smart(img_data, original_dpi=None):
    if isinstance(img_data, bytes):
        original_bytes_size = len(img_data)
        img = Image.open(io.BytesIO(img_data))
    else:
        img = img_data
        tmp = io.BytesIO()
        img.save(tmp, format=img.format or "PNG")
        original_bytes_size = len(tmp.getvalue())

    # Force decoding once
    img.load()
    original_img = img  
    width, height = img.size


    use_simple_jpeg = (
        original_bytes_size < CompressionConfig.MIN_SIZE_FOR_PCA
        or min(width, height) < CompressionConfig.MIN_DIMENSION_FOR_PCA
    )

    
    n_components = CompressionConfig.IMAGE_COMPONENTS
    min_variance = CompressionConfig.IMAGE_MIN_VARIANCE
    jpeg_quality = CompressionConfig.IMAGE_JPEG_QUALITY

    # === SIMPLE JPEG PATH (for small images) ===
    if use_simple_jpeg:
        try:
            print("here in simple jpeg")
            compressed_bytes = simple_jpeg_compress(
                original_img, jpeg_quality, dpi=original_dpi
            )

            return compressed_bytes

        except Exception as e:
            print("in simple error ")
            # Return original on error
            if isinstance(img_data, bytes):
                return img_data
            else:
                tmp = io.BytesIO()
                original_img.save(tmp, format=original_img.format or "PNG")
                raw_bytes = tmp.getvalue()
                return raw_bytes

    # === PCA PATH (for larger images) ===
    try:
        print("here in pca")
        # Create a separate image for PCA operations to avoid mutating original
        pca_img = img

        # Convert to RGB if needed for PCA
        if pca_img.mode in ("RGBA", "LA", "P"):
            pca_img = pca_img.convert("RGB")

        # Optional downsampling for very large images
        max_dimension = CompressionConfig.IMAGE_MAX_SIZE
        if max(pca_img.size) > max_dimension:
            print("downsapling a large img")
            scale = max_dimension / max(pca_img.size)
            new_size = (int(pca_img.width * scale), int(pca_img.height * scale))
            # using resampling filter lanczos from Pillow library
            pca_img = pca_img.resize(new_size, Image.Resampling.LANCZOS)

        img_array = np.array(pca_img)

        compressor = PCAImageCompressor()

        # PCA compression
        compressed_array, _, variance = compressor.compress_image(
            img_array,
            max_components=n_components,
            min_variance=min_variance,
        )

        # Convert back to image
        if compressed_array.ndim == 2:
            compressed_img = Image.fromarray(compressed_array, mode="L")
        else:
            compressed_img = Image.fromarray(compressed_array, mode="RGB")

        # Save as JPEG
        output = io.BytesIO()
        save_kwargs = {
            "format": "JPEG",
            "quality": jpeg_quality,
            "optimize": True,
        }
        if original_dpi is not None:
            save_kwargs["dpi"] = (int(original_dpi), int(original_dpi))

        compressed_img.save(output, **save_kwargs)
        compressed_bytes = output.getvalue()
        compressed_size = len(compressed_bytes)

        # Check if PCA helped enough (at least 5% saving)
        if compressed_size >= original_bytes_size * 0.95:
            # Fall back to simple JPEG on original image (no PCA downscale)
            compressed_bytes = simple_jpeg_compress(
                original_img, jpeg_quality, dpi=original_dpi
            )
            compressed_size = len(compressed_bytes)

        # Final check: if still bigger, return original
        if compressed_size >= original_bytes_size:
            if isinstance(img_data, bytes):
                compressed_bytes = img_data
            else:
                tmp = io.BytesIO()
                original_img.save(tmp, format=original_img.format or "PNG")
                compressed_bytes = tmp.getvalue()
            compressed_size = len(compressed_bytes)

        return compressed_bytes

    except Exception as e:
        print("in pca error")
        # Final fallback: simple JPEG or original
        try:
            compressed_bytes = simple_jpeg_compress(
                original_img, jpeg_quality, dpi=original_dpi
            )

            return compressed_bytes
        except Exception:
            # Absolute last resort: return original
            if isinstance(img_data, bytes):
                return img_data
            else:
                tmp = io.BytesIO()
                original_img.save(tmp, format=original_img.format or "PNG")
                raw_bytes = tmp.getvalue()
                return raw_bytes
