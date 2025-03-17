import os
import pytesseract
import time

# Set correct paths for Ubuntu
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata/'
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Test OCR
start = time.time()
print(pytesseract.image_to_string('sample_page.jpg', lang='eng1+eng'))
end = time.time()
print(f"OCR Time: {end - start:.2f}s")