from ultralytics import YOLO
model = YOLO('/datasets/runs/detect/yolov8_text_seg3/weights/best.pt')
model.to('cuda:7')
metrics = model.val(data='/datasets/data.yaml')
print(metrics.box.map)  # Print mean Average Precision
