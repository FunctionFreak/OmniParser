import os
import base64
import io
from PIL import Image
import torch
from typing import Tuple

from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

def process_image(
    image_path: str,
    box_threshold: float = 0.05,
    iou_threshold: float = 0.1,
    use_paddleocr: bool = True,
    imgsz: int = 640
) -> Tuple[Image.Image, str]:
    """
    Process a single image and return the labeled image and parsed content.
    """
    # Load image
    image_input = Image.open(image_path)
    
    box_overlay_ratio = image_input.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_input, 
        display_img=False, 
        output_bb_format='xyxy', 
        goal_filtering=None, 
        easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
        use_paddleocr=use_paddleocr
    )
    text, ocr_bbox = ocr_bbox_rslt
    
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_input, 
        yolo_model, 
        BOX_TRESHOLD=box_threshold, 
        output_coord_in_ratio=True, 
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config, 
        caption_model_processor=caption_model_processor, 
        ocr_text=text,
        iou_threshold=iou_threshold, 
        imgsz=imgsz
    )
    
    # Convert base64 image to PIL Image
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    
    # Format parsed content
    parsed_content_text = '\n'.join([f'icon {i}: ' + str(v) for i, v in enumerate(parsed_content_list)])
    
    return image, parsed_content_text

def main():
    """
    Main function to process all images in the test/ folder.
    """
    # Load models
    print("Loading models...")
    global yolo_model, caption_model_processor
    yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
    caption_model_processor = get_caption_model_processor(
        model_name="blip2", 
        model_name_or_path="Salesforce/blip2-opt-2.7b"
    )
    print("Models loaded successfully!")
    
    # Create directories if they don't exist
    os.makedirs('test', exist_ok=True)
    os.makedirs('result', exist_ok=True)
    
    # Get all image files from test directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    test_dir = 'test'
    result_dir = 'result'
    
    image_files = []
    for file in os.listdir(test_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if not image_files:
        print("No image files found in test/ directory!")
        print("Please add some images (.jpg, .jpeg, .png, .bmp, .tiff, .webp) to the test/ folder.")
        return
    
    print(f"Found {len(image_files)} image(s) to process:")
    for img_file in image_files:
        print(f"  - {img_file}")
    
    # Process each image
    for image_file in image_files:
        try:
            print(f"\nProcessing {image_file}...")
            image_path = os.path.join(test_dir, image_file)
            
            # Process the image
            processed_image, parsed_content = process_image(image_path)
            
            # Save processed image
            base_name = os.path.splitext(image_file)[0]
            output_image_path = os.path.join(result_dir, f"{base_name}_processed.png")
            processed_image.save(output_image_path)
            
            # Save parsed content as text file
            output_text_path = os.path.join(result_dir, f"{base_name}_results.txt")
            with open(output_text_path, 'w', encoding='utf-8') as f:
                f.write(f"Results for {image_file}\n")
                f.write("=" * 50 + "\n\n")
                f.write(parsed_content)
            
            print(f"✓ Processed {image_file}")
            print(f"  - Saved processed image: {output_image_path}")
            print(f"  - Saved results text: {output_text_path}")
            
        except Exception as e:
            print(f"✗ Error processing {image_file}: {str(e)}")
            continue
    
    print(f"\nDone! Check the result/ folder for processed images and text results.")

if __name__ == "__main__":
    main() 