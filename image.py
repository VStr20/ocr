import time
import cv2
from ultralytics import YOLO
import pytesseract
from pdf2image import convert_from_path

# --- PDF to Image Extraction ---
# Path to the PDF file
pdf_path = 'attn_is_all_you_need.pdf'

# Convert the first page of the PDF to an image with 300 dpi (adjust dpi as needed)
pages = convert_from_path(pdf_path, dpi=300)

# Save the first page as an image
image_path = 'sample_page.jpg'
pages[1].save(image_path, 'JPEG')
print(f"Page extracted and saved as {image_path}")

# --- Layout Segmentation using YOLOv8 ---
# Load YOLOv8 model (nano variant for faster performance)
model = YOLO('yolov8n.pt')

# Load the extracted image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Time the segmentation process
start_time = time.time()

# Perform layout segmentation using YOLOv8 to detect text blocks
results = model(image)

end_time = time.time()
inference_time = end_time - start_time
print(f"YOLOv8 Inference Time: {inference_time:.2f} seconds")

# --- OCR on Detected Text Blocks ---
# This example assumes that the YOLO model detects text blocks and provides bounding boxes.
# The following code iterates over detected boxes and applies pytesseract to each cropped region.
print("OCR results for detected text blocks:")
for result in results:
    # Check if the results contain bounding boxes. Adjust according to your YOLOv8 output structure.
    if hasattr(result, "boxes") and result.boxes is not None:
        for box in result.boxes.xyxy:  # each box is in [x1, y1, x2, y2] format
            x1, y1, x2, y2 = map(int, box)
            # Crop the detected region of interest (ROI) from the image
            roi = image[y1:y2, x1:x2]
            # Optionally, you can preprocess ROI (grayscale, thresholding, etc.) for better OCR accuracy
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Use pytesseract to extract text from the ROI
            ocr_text = pytesseract.image_to_string(gray_roi, config='--oem 3 --psm 6')
            print("Detected Text:", ocr_text)
