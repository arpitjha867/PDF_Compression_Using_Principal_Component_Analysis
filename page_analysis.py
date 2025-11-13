import fitz 
import json
import os
from PIL import Image
from PIL import ImageDraw, ImageFont 



def extract_page_elements(page):
    regions = []
    
    text_blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, ...)
    
    for block in text_blocks:
        x0, y0, x1, y1 = block[:4]
        text_content = block[4] if len(block) > 4 else ""
        #skip white space
        if not text_content.strip():
            continue
        
        regions.append({
            "type": "text",
            "bbox": [float(x0), float(y0), float(x1), float(y1)],
        })
    
    # images
    image_list = page.get_images(full=True)
    
    for img_index, img_info in enumerate(image_list):
        xref = img_info[0]  # Image reference number
        
        # Get image position on page
        img_rects = page.get_image_rects(xref)
        
        for rect in img_rects:
            x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
            
            regions.append({
                "type": "image",
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "xref": xref
            })
    
    return regions

def draw_annotations(page, regions, output_path):
    # page -> img
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    draw = ImageDraw.Draw(img)
    
    scale = 2
    
    for region in regions:
        x0, y0, x1, y1 = region['bbox']
        x0, y0, x1, y1 = x0 * scale, y0 * scale, x1 * scale, y1 * scale
        
        if region['type'] == 'text':
                   # R   G   B
            color = (0, 255, 0)  
            label = "TEXT"
        elif region['type'] == 'image':
            color = (255, 0, 0) 
            label = "IMAGE"
        else:
            color = (128, 128, 128)  
            label = region['type'].upper()
        
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        
        draw.text((x0 + 5, y0 + 5), label, fill=color)
    
    img.save(output_path, "PNG")
    print(f" Saved annotated image: {output_path}")


def process_pdf(input_pdf_path, output_images_folder):
    doc = fitz.open(input_pdf_path)
    all_metadata = {}
    
    print(f"Processing PDF: {input_pdf_path}")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        print(f"[Page {page_num + 1}]")
        
        regions = extract_page_elements(page)
        
        page_name = f"page_{page_num + 1:03d}.json"
        all_metadata[page_name] = regions
        
        img_path = os.path.join(output_images_folder, f"annotated_page_{page_num + 1:03d}.png")
        draw_annotations(page, regions, img_path)
    
    doc.close()
    

    json_path = os.path.join(output_images_folder, "region_metadata.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f" Processing complete!")
    
    return all_metadata