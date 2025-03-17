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

os.environ["OMP_THREAD_LIMIT"] = "1"
cv2.setNumThreads(0)  # Disable OpenCV threading

NUM_WORKERS = multiprocessing.cpu_count() - 1
# print(NUM_WORKERS)

# yolo_model = YOLO('/home/sunny/CCRL_vs/ocr/datasets/runs/detect/train5/weights/best.pt')
yolo_model = YOLO('yolov8n.pt')
yolo_model.fuse()  # Fuse Conv+BN layers for faster inference

yolo_model.to('cuda:7')

dummy_img = np.zeros((1100, 1000, 3), dtype=np.uint8)
_ = yolo_model(dummy_img)

# def resize_image(image, max_dimension=MAX_IMAGE_DIMENSION):
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
                                     cv2.THRESH_BINARY, 9, 2)
        # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    #  cv2.THRESH_BINARY, 11, 2)
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # kernel = np.ones((1, 1), np.uint8)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary

# def extract_text_blocks(image):
#     # Handle large images
#     # h, w = image.shape[:2]
#     # if max(h, w) > 1280:
#     #     scale = 1280 / max(h, w)
#     #     image = cv2.resize(image, (int(w*scale), int(h*scale)))
    
#     # Run inference
#     results = yolo_model.predict(
#         image,
#         imgsz=640,
#         conf=0.3,  # Optimal balance for document text
#         iou=0.3,
#         device=yolo_model.device,
#         verbose=False
#     )[0]
    
#     # Process results
#     text_blocks = []
#     class_names = yolo_model.names
    
#     if results.boxes:
#         boxes = results.boxes.xyxy.cpu().numpy()
#         class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
#         for box, cls_id in zip(boxes, class_ids):
#             if class_names[cls_id] == "item":
#                 x1, y1, x2, y2 = map(int, box)
#                 if x2 > x1 and y2 > y1:
#                     text_blocks.append(image[y1:y2, x1:x2])
    
#     return text_blocks

def extract_text_blocks(image):
    
    device = yolo_model.device
    
    results = yolo_model.predict(image, verbose=False, imgsz=640, half=True, conf=0.25, iou=0.35,  device=device)[0]
    
    # results = yolo_model(image, verbose=False, conf=0.55, iou=0.55, device=device)[0]
    # results.show()

    text_blocks = []
    if hasattr(results, 'masks') and results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()

        for mask, box in zip(masks, boxes):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cropped_img = image[y1:y2, x1:x2]
            text_blocks.append(cropped_img)
    else:
        text_blocks.append(image)

    return text_blocks

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
        
def ocr_pdf_page(pdf_path, page_num=1):
    start_time = time.time()
    
    start1 = time.time()
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
    
    # for i, block in enumerate(text_blocks):
    #     plt.figure(figsize=(3, 3))
    #     plt.imshow(cv2.cvtColor(block, cv2.COLOR_BGR2RGB))
    #     plt.title(f"Block {i+1}")
    #     plt.axis('off')
    # plt.show()
    
    
    end2 = time.time()
    print(f"Text Block Extraction Time: {end2 - start2:.3f} seconds")
    
    start3 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        print(len(text_blocks), NUM_WORKERS)
        chunk_size = (len(text_blocks) // (NUM_WORKERS))
        results = list(executor.map(process_text_block, text_blocks, chunksize=chunk_size))
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
    
    print(f"\n Processing completed in {processing_time:.3f} seconds.")
    
    file = open("observed_output.txt", "w") 

    for i in range(3): 
        file.write(extracted_text)         
    file.close() 
    
    if args.ground_truth:
        with open(args.ground_truth) as f:
            ground_truth_text = f.read()
        
        accuracy_percentage = test_accuracy(extracted_text, ground_truth_text)
        
    print(f"\n Accuracy: {accuracy_percentage:.2f}%")
