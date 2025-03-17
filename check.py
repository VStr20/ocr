import os
import cv2
import numpy as np
from pdf2image import convert_from_path
from ultralytics import YOLO
import pytesseract
from PIL import Image
import concurrent.futures
import multiprocessing
import time
import matplotlib.pyplot as plt

os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata/'
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Optional: Use tesserocr if installed (faster than pytesseract)
# try:
#     from tesserocr import PyTessBaseAPI, PSM
#     USE_TESSEROCR = True
# except ImportError:
#     USE_TESSEROCR = False

# Disable internal threading in Tesseract to avoid contention
os.environ["OMP_THREAD_LIMIT"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
cv2.setNumThreads(0)  # Disable OpenCV threading

# Global configurations
NUM_WORKERS = multiprocessing.cpu_count() - 1
# print(NUM_WORKERS)
# MAX_IMAGE_DIMENSION = 1500  # pixels

# Load YOLOv8 segmentation model (lightweight version)
yolo_model = YOLO('datasets/runs/detect/train5/weights/best.pt')
yolo_model.fuse()  # Fuse Conv+BN layers for faster inference
# yolo_model = YOLO('yolov5nu.pt')
yolo_model.to('cuda:7')
print(yolo_model.names)  # See actual class names

dummy_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
_ = yolo_model(dummy_img)

# def resize_image_if_needed(image, max_dimension=MAX_IMAGE_DIMENSION):
#     height, width = image.shape[:2]
#     if max(height, width) <= max_dimension:
#         return image

#     scale_factor = max_dimension / max(height, width)
#     new_size = (int(width * scale_factor), int(height * scale_factor))
#     return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

def preprocess_image(image):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    h, w = gray.shape
    if max(h, w) > 1500:
        scale = 1500 / max(h, w)
        gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    if np.mean(gray) < 16:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    #  cv2.THRESH_BINARY, 11, 2)
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # kernel = np.ones((1, 1), np.uint8)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary

def preprocess_crop(crop):
    """Enhances OCR readability"""
    # Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Denoise and sharpen
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Adaptive thresholding
    return cv2.adaptiveThreshold(sharpened, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def nearest_in_list(number, list_of_numbers):
    """Helper function for grid snapping"""
    closest = min(list_of_numbers, key=lambda x: abs(x-number))
    candidates = [n for n in list_of_numbers if n >= number]
    return min(candidates) if candidates else closest

# Visualization helper (for debugging)
# def visualize_blocks(image, text_blocks):
#     """Helper to visualize ordered text blocks"""
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(12, 8))
#     # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
#     for i, block in enumerate(text_blocks):
#         plt.figure(figsize=(2, 2))
#         # plt.imshow(cv2.cvtColor(block, cv2.COLOR_BGR2RGB))
#         plt.title(f"Block {i+1}")
#         plt.axis('off')
#     plt.savefig('text_blocks.png')
#     plt.show()
    
def extract_text_blocks(image):
    # Initialize model and get device
    device = yolo_model.device
    
    results = yolo_model.predict(
        image,
        imgsz=640,
        conf=0.4,  # Increased confidence threshold
        iou=0.4,   # More precise box selection
        half=True,
        device=device,
        verbose=False
    )[0]

    text_blocks = []
    boxes_info = []
    h, w = image.shape[:2]

    # Grid parameters (adjust based on document layout)
    grid_size = 16  # Matching the example's 22px grid
    vertical_grid = np.arange(0, h, grid_size).tolist()
    horizontal_grid = np.arange(0, w, grid_size).tolist()

    if results.boxes:
        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        # Process detected boxes
        for box, cls_id in zip(boxes, class_ids):
            if yolo_model.names[cls_id] == "item":  # Verify your class name
                x1, y1, x2, y2 = map(int, box)
                if x2 > x1 and y2 > y1 and (x2-x1) > 5 and (y2-y1) > 5:
                    # Snap coordinates to grid
                    snapped_y = nearest_in_list(y1, vertical_grid)
                    snapped_x = nearest_in_list(x1, horizontal_grid)
                    
                    boxes_info.append({
                        'coords': (x1, y1, x2, y2),
                        'snapped': (snapped_x, snapped_y),
                        'crop': image[y1:y2, x1:x2]
                    })

        # Sort blocks using grid-based ordering
        if boxes_info:
            # Create sorting key: vertical first, then horizontal
            boxes_info.sort(key=lambda b: (b['snapped'][1], b['snapped'][0]))
            
            # Extract ordered crops
            text_blocks = [b['crop'] for b in boxes_info]
            
    text_blocks = [preprocess_for_ocr(block) for block in text_blocks]

    return text_blocks

def preprocess_for_ocr(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

def extract_text_blocks2(image):
    results = yolo_model(
        image,
        imgsz=640,
        conf=0.4,  # Increased confidence
        iou=0.4,   # Tighter overlap
        device=yolo_model.device,
        verbose=False
    )[0]

    text_blocks = []
    h, w = image.shape[:2]

    if results.boxes:
        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        valid_boxes = []
        for box, cls_id in zip(boxes, class_ids):
            # Verify class name matches your model
            if yolo_model.names[cls_id] == "item":  # ✅
                x1, y1, x2, y2 = map(int, box)
                if x2 > x1 and y2 > y1 and (x2-x1) > 5 and (y2-y1) > 5:
                    valid_boxes.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'crop': image[y1:y2, x1:x2]
                    })

        if valid_boxes:
            # Calculate dynamic parameters
            line_heights = [b['y2']-b['y1'] for b in valid_boxes]
            line_height = int(np.mean(line_heights)) if line_heights else 30
            col_margin = 0.02 * w  # 2% of image width

            # Line grouping
            lines = []
            current_line = []
            last_bottom = -float('inf')
            
            for box in sorted(valid_boxes, key=lambda b: b['y1']):
                if box['y1'] > last_bottom + line_height//2:
                    if current_line:
                        lines.append(current_line)
                    current_line = [box]
                    last_bottom = box['y2']
                else:
                    current_line.append(box)
                    last_bottom = max(last_bottom, box['y2'])
            if current_line:
                lines.append(current_line)

            # Column detection and sorting
            sorted_blocks = []
            for line in lines:
                # Sort line left-to-right
                line = sorted(line, key=lambda b: b['x1'])
                
                # Column grouping
                columns = []
                current_col = []
                last_right = -float('inf')
                
                for box in line:
                    if box['x1'] > last_right + col_margin:
                        if current_col:
                            columns.append(current_col)
                        current_col = [box]
                        last_right = box['x2']
                    else:
                        current_col.append(box)
                        last_right = max(last_right, box['x2'])
                if current_col:
                    columns.append(current_col)
                
                # Sort columns left-to-right
                for col in sorted(columns, key=lambda c: c[0]['x1']):
                    sorted_blocks.extend(col)

            # Merge and preprocess
            merged_blocks = []
            for block in sorted_blocks:
                processed = preprocess_crop(block['crop'])
                merged_blocks.append(processed)
            
            text_blocks = merged_blocks
        
        text_blocks = [preprocess_for_ocr(block) for block in text_blocks]

    return text_blocks

def extract_text_blocks1(image):
    results = yolo_model(
        image,
        imgsz=1280,
        conf=0.25,
        iou=0.25,
        device=yolo_model.device,
        verbose=False
    )[0]

    text_blocks = []
    h, w = image.shape[:2]
    
    if valid_boxes:
        line_heights = [b['y2']-b['y1'] for b in valid_boxes]
        line_height = int(np.mean(line_heights)) if line_heights else 30
    col_margin = 0.02 * w  # 2% of image width
    
    if results.boxes:
        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        valid_boxes = []
        for box, cls_id in zip(boxes, class_ids):
            if yolo_model.names[cls_id] == "item":  # Verify your class name
                x1, y1, x2, y2 = map(int, box)
                if x2 > x1 and y2 > y1:
                    valid_boxes.append({
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'crop': image[y1:y2, x1:x2]
                    })

        if valid_boxes:
            lines = []
            current_line = []
            last_bottom = -float('inf')
            
            for box in sorted(valid_boxes, key=lambda b: b['y1']):
                if box['y1'] > last_bottom + line_height//2:
                    if current_line:
                        lines.append(current_line)
                    current_line = [box]
                    last_bottom = box['y2']
                else:
                    current_line.append(box)
                    last_bottom = max(last_bottom, box['y2'])
            if current_line:
                lines.append(current_line)

            sorted_blocks = []
            for line in lines:
                # Split into columns if needed
                columns = []
                current_col = []
                last_right = -float('inf')
                
                for box in sorted(line, key=lambda b: b['x1']):
                    if box['x1'] > last_right + col_margin:
                        if current_col:
                            columns.append(current_col)
                        current_col = [box]
                        last_right = box['x2']
                    else:
                        current_col.append(box)
                        last_right = max(last_right, box['x2'])
                if current_col:
                    columns.append(current_col)
                
                sorted_line = []
                for col in sorted(columns, key=lambda c: c[0]['x1']):
                    sorted_line.extend(sorted(col, key=lambda b: b['x1']))
                
                sorted_blocks.extend(sorted_line)

            text_blocks = [block['crop'] for block in sorted_blocks]

    return text_blocks

# def extract_text_blocks(image):
#     results = yolo_model(
#         image,
#         imgsz=1280,
#         conf=0.25,
#         iou=0.25,
#         device=yolo_model.device,
#         verbose=False
#     )[0]
    
#     # show_results(results)
    
#     text_blocks = []
#     boxes_info = []
#     class_names = yolo_model.names
#     h, w = image.shape[:2]

#     if results.boxes:
#         boxes = results.boxes.xyxy.cpu().numpy()
#         class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
#         for box, cls_id in zip(boxes, class_ids):
#             if class_names[cls_id] == "item":  # Verify your class name
#                 x1, y1, x2, y2 = map(int, box)
#                 if x2 > x1 and y2 > y1:
#                     boxes_info.append({
#                         'coords': (x1, y1, x2, y2),
#                         'crop': image[y1:y2, x1:x2]
#                     })

#         if boxes_info:
#             grid_y = 5
#             vertical_grid = np.arange(0, h, grid_y)
            
#             grid_x = 5
#             horizontal_grid = np.arange(0, w, grid_x)
            
#             sorted_blocks = []
#             for box in boxes_info:
#                 x1, y1, x2, y2 = box['coords']
#                 center_y = (y1 + y2) // 2
#                 center_x = (x1 + x2) // 2
                
#                 snap_y = min(vertical_grid, key=lambda y: abs(y - center_y))
#                 snap_x = min(horizontal_grid, key=lambda x: abs(x - center_x))
                
#                 sorted_blocks.append((
#                     snap_y,  # Primary sort by vertical position
#                     snap_x,  # Secondary sort by horizontal position
#                     box['crop']  # Keep reference to the crop
#                 ))

#             sorted_blocks.sort(key=lambda x: (x[0], x[1]))
            
#             text_blocks = [block[2] for block in sorted_blocks]

#     return text_blocks

def process_text_block(block_img):
    if block_img.shape[0] < 10 or block_img.shape[1] < 20:
        return ""

    processed_img = preprocess_image(block_img)
    # config = (
    #     '--psm 6 '            # Assume single uniform block
    #     '--oem 1 '             # LSTM only (fastest)
    #     '-c tessedit_do_invert=0 '  # No inversion check
    #     '-c preserve_interword_spaces=0 '  # Reduce spacing analysis
    #     '-c textord_heavy_nr=1 '     # Less noise reduction
    #     '-c textord_min_linesize=2.0 '  # Accept smaller text
    # )
    config = '--psm 6 --oem 1'  # Default config
    # config = '--psm 6 --oem 1' # LSTM only mode (faster than default oem=3)
    # config = '--psm 6 --oem 1 -l eng'
    
    text = pytesseract.image_to_string(processed_img, config=config, lang='eng')
    return text.strip()


def show_results(results):
    """Visualize results using matplotlib"""
    for r in results:
        # Convert BGR to RGB and create plot
        im_array = r.plot()  # Get BGR numpy array
        im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        
        # Create matplotlib figure
        plt.figure(figsize=(12, 8))
        plt.imshow(im_rgb)
        plt.axis('off')  # Hide axes
        plt.savefig('text_boxes.png')
        plt.close()
        
def ocr_pdf_page(pdf_path, page_num=1):
    start_time = time.time()
    
    start1 = time.time()
    # images = convert_from_path(pdf_path, dpi=150, thread_count=NUM_WORKERS)
    images = convert_from_path(pdf_path,
                               first_page=page_num+1,
                               last_page=page_num+1,
                               dpi=90, thread_count=NUM_WORKERS)
                            #    grayscale=True)
    # images = convert_from_path(pdf_path,
    #                        first_page=page_num+1,
    #                        last_page=page_num+1,
    #                        dpi=90,
    #                        thread_count=4,  # Optimal for most CPUs
    #                        grayscale=True,  # Direct grayscale conversion
    #                        use_pdftocairo=True,  # Faster rendering
    #                        fmt='jpeg',  # Faster than PNG
    #                        jpegopt={"quality": 85, "progressive": True})
    end1 = time.time()
    print(f"PDF to Image Conversion Time: {end1 - start1:.3f} seconds")

    start5 = time.time()
    image_cv = np.array(images[0])
    
    lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    limg = cv2.merge([clahe.apply(l), a, b])
    image_cv = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    end5 = time.time()
    print(f"CLAHE Time: {end5 - start5:.3f} seconds")

    # print(image_cv.shape)
    # image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    start2 = time.time()
    text_blocks = extract_text_blocks(image_cv)
    
    # For visualization:
    # visualize_blocks(images, text_blocks)
    
    # for i, block in enumerate(text_blocks):
    #     plt.figure(figsize=(3, 3))
    #     plt.imshow(cv2.cvtColor(block, cv2.COLOR_BGR2RGB))
    #     plt.title(f"Block {i+1}")
    #     plt.axis('off')
    # plt.show()
    
    
    # vis_data = extract_text_blocks1(image_cv)
    # visualize_text_blocks(vis_data)  # Save visualization
    # print(len(text_blocks))
    end2 = time.time()
    print(f"Text Block Extraction Time: {end2 - start2:.3f} seconds")
    
    start3 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        print(len(text_blocks), NUM_WORKERS)
        chunk_size = (len(text_blocks) // (NUM_WORKERS))
        results = list(executor.map(process_text_block, text_blocks, chunksize=chunk_size))
        # results = list(executor.map(process_text_block, text_blocks))
    end3 = time.time()
    print(f"Text Block Processing Time: {end3 - start3:.3f} seconds")
    
    start4 = time.time()
    full_text = "\n\n".join(filter(None, results))
    end4 = time.time()
    print(f"Text Concatenation Time: {end4 - start4:.3f} seconds")
    
    elapsed_time = time.time() - start_time
    
    return full_text.strip(), elapsed_time

def test_accuracy(extracted_text, ground_truth):
    """
    Compare extracted OCR text against ground truth to measure accuracy.
    
    Args:
        extracted_text (str): Text extracted by OCR.
        ground_truth (str): Expected ground truth text.
    
    Returns:
        float: Accuracy as percentage.
    """
    extracted_words = set(extracted_text.split())
    ground_truth_words = set(ground_truth.split())
    
    common_words = extracted_words.intersection(ground_truth_words)
    
    accuracy_percentage = (len(common_words) / len(ground_truth_words)) * 100 if ground_truth_words else 0.0
    
    return accuracy_percentage

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimized OCR using YOLOv8 and Multi-threading")
    parser.add_argument('pdf_path', type=str, help='Path to the PDF file')
    parser.add_argument('--page', type=int, default=0, help='Page number to process (starting from 0)')
    parser.add_argument('--ground_truth', type=str,
                        help="Path to a text file containing ground truth for accuracy testing")

    args = parser.parse_args()

    # print(f"Processing '{args.pdf_path}', page {args.page + 1}...")
    
    extracted_text, processing_time = ocr_pdf_page(args.pdf_path, args.page)
    
    file = open("observed_output.txt", "w") 

    for i in range(3): 
        file.write(extracted_text)         
    file.close() 
    
    print(f"\n Processing completed in {processing_time:.3f} seconds.")
    
    if args.ground_truth:
        with open(args.ground_truth) as f:
            ground_truth_text = f.read()
        
        accuracy_percentage = test_accuracy(extracted_text, ground_truth_text)
        
    print(f"\n Accuracy: {accuracy_percentage:.2f}%")
