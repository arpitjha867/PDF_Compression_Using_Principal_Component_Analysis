import fitz
import os
from helper_function import compress_image_smart


def compress_pdf_smart(input_path, output_path, metadata = None):
    doc = fitz.open(input_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        print(f" Page {page_num + 1}/{len(doc)}")

        image_list = page.get_images(full=True)
        if not image_list:
            print(" No images found on this page\n")
            continue
        # this whole loop says taht we are compressing images only no text
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]

                compressed_bytes = compress_image_smart(
                    img_bytes,
                    # region_type,
                    original_dpi=base_image.get("dpi", (72, 72))[0],
                )

                page.replace_image(xref, stream=compressed_bytes)

            except Exception as e:
                print(f"  Image {img_index+1}: Error - {e}")
        page.clean_contents(sanitize=True)
        print()

    # Save compressed PDF
    # print(" Saving compressed PDF...")
    doc.save(
        output_path,
        garbage=4,
        deflate=True,
        clean=True,
        linear=False,
    )
    doc.close()
