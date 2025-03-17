import concurrent.futures
import pytesseract
import cv2
import numpy as np
import torch
import torchvision
from pdf2image import convert_from_path
from ultralytics import YOLO
import time
import os

os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata/'
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def convert_pdf_to_image(pdf_path: str, dpi: int = 300) -> np.ndarray:
    """Converts the first page of a PDF to a numpy image array."""
    images = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)
    if not images:
        raise ValueError("No pages found in the PDF.")
    return cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)

def detect_text_blocks(image: np.ndarray) -> list:
    """Detects text blocks using a YOLOv8 model trained for document layout segmentation."""
    
    model = YOLO("yolov8n-seg.pt")  # Replace with a document layout model if available
    
    # model_path = '/home/sunny/CCRL_vs/ocr/readingletters-yolov8l-augv1.pt'  # Replace with actual path
    # model = YOLO(model_path)

    # # Check model summary
    # model.info()
    model.to("cuda:2")  # Use GPU for faster inference
    
    # results = model("/home/sunny/CCRL_vs/ocr/sample_page.jpg")  # Replace with actual image path

    # for result in results:
    #     print(result)
    # results = model.predict(source=image, conf=0.25)  # Adjust conf threshold as needed
    # print(f"Results after detection", results)
    results = model(image, verbose=False)
    # print(f"Results after detection", results)
    text_boxes = []
    for result in results:
        for box in result.boxes:
            if model.names[int(box.cls)] == "text":  # Adjust class name based on your model
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                text_boxes.append((x1, y1, x2, y2))
    return text_boxes

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocesses the image for OCR (grayscale + thresholding)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

def ocr_processing(crop: np.ndarray) -> str:
    """Processes a single text block with Tesseract OCR."""
    processed_crop = preprocess_image(crop)
    text = pytesseract.image_to_string(
        processed_crop,
        config="--psm 6 --oem 1",  # Single block, LSTM-only
    )
    return text.strip()

def process_pdf(pdf_path: str) -> str:
    start_time = time.time()

    # Step 1: Convert PDF to image
    image = convert_pdf_to_image(pdf_path, dpi=300)
    print(f"PDF conversion time: {time.time() - start_time:.2f}s")

    print(f"Image after pdf to image", image)
    
    # Step 2: Detect text blocks using YOLOv8
    text_boxes = detect_text_blocks(image)
    print(f"YOLOv8 detection time: {time.time() - start_time:.2f}s")
    
    print(f"Text boxes after detection", text_boxes)

    if not text_boxes:
        return "No text blocks detected."

    # Step 3: Sort text blocks by vertical position (top to bottom)
    text_boxes.sort(key=lambda box: box[1])

    # Step 4: Crop text regions
    crops = []
    for box in text_boxes:
        x1, y1, x2, y2 = box
        crops.append(image[y1:y2, x1:x2])

    # Step 5: Parallel OCR processing
    final_text = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(ocr_processing, crop) for crop in crops]
        for future in concurrent.futures.as_completed(futures):
            final_text.append(future.result())

    # Combine results in reading order
    combined_text = "\n".join(final_text)
    print(f"Total processing time: {time.time() - start_time:.2f}s")
    return combined_text

# Example usage
if __name__ == "__main__":
    pdf_path = "attn_is_all_you_need.pdf"  # Replace with your PDF path
    extracted_text = process_pdf(pdf_path)
    print(extracted_text)
