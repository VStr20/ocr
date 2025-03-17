from ultralytics import YOLO
model = YOLO('/home/sunny/CCRL_vs/ocr/datasets/runs/detect/yolov8_text_seg3/weights/best.pt')
model.to('cuda:7')
metrics = model.val(data='/home/sunny/CCRL_vs/ocr/datasets/data.yaml')
print(metrics.box.map)  # Print mean Average Precision