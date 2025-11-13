import fitz
import os
from helper_function import compress_image_smart


def compress_pdf_smart(input_path, output_path, metadata):
    """Compress PDF using PCA with region metadata."""
    doc = fitz.open(input_path)
    original_size = os.path.getsize(input_path)

    total_stats = {
        "text": {"count": 0, "original": 0, "compressed": 0, "variance": []},
        "image": {"count": 0, "original": 0, "compressed": 0, "variance": []},
        "unknown": {"count": 0, "original": 0, "compressed": 0, "variance": []},
    }

    for page_num in range(len(doc)):
        page = doc[page_num]
        print(f"📄 Page {page_num + 1}/{len(doc)}")

        page_key = f"page_{page_num + 1:03d}.json"
        if page_key not in metadata:
            print("   ⚠ No metadata found for this page\n")
            continue

        regions = metadata[page_key]
        image_list = page.get_images(full=True)
        if not image_list:
            print("   ℹ No images found on this page\n")
            continue

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]

            # Lookup region type using metadata xref
            region_type = next(
                (r["type"] for r in regions if r.get("xref") == xref), "unknown"
            )

            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]

                compressed_bytes, stats = compress_image_smart(
                    img_bytes,
                    region_type,
                    original_dpi=base_image.get("dpi", (72, 72))[0],
                )

                page.replace_image(xref, stream=compressed_bytes)

                total_stats[region_type]["count"] += 1
                total_stats[region_type]["original"] += stats["original_size"]
                total_stats[region_type]["compressed"] += stats["compressed_size"]
                if stats["variance_explained"] > 0:
                    total_stats[region_type]["variance"].append(
                        stats["variance_explained"]
                    )

                print(
                    f"   Image {img_index+1}: {region_type:8s} | "
                    f"{stats['components_used']:3d} comp | "
                    f"{stats['variance_explained']:5.1f}% var | "
                    f"{stats['compression_ratio']:4.1f}x ratio"
                )

            except Exception as e:
                print(f"   ❌ Image {img_index+1}: Error - {e}")

        print()

    # Save compressed PDF
    print("💾 Saving compressed PDF...")
    doc.save(
        output_path,
        garbage=4,
        deflate=True,
        clean=True,
        linear=False,
    )
    doc.close()

    compressed_size = os.path.getsize(output_path)
    savings = original_size - compressed_size
    compression_pct = (1 - compressed_size / original_size) * 100

    print("\n" + "="*50)
    print("📊 COMPRESSION REPORT")
    print("="*50)
    print(f"Original size:   {original_size/1024/1024:.2f} MB")
    print(f"Compressed size: {compressed_size/1024/1024:.2f} MB")
    print(f"Saved:           {savings/1024/1024:.2f} MB ({compression_pct:.1f}% smaller)")
    print("="*50 + "\n")

    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "savings": savings,
        "compression_percentage": compression_pct,
        "stats_by_type": total_stats
    }