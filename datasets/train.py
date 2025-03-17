from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load model
model.to('cuda:7')  # Move to GPU
model.train(
    data='data.yaml',
    epochs=360,
    patience=0,  # Stop if no improvement for 50 epochs
    batch=8,
    imgsz=640,  # Critical for small text
    close_mosaic=10,  # Disable mosaic in final epochs
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=3,
    mixup=0.15,
    copy_paste=0.3,  # Text-specific augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=15,
    translate=0.1,
    scale=0.9,
    shear=5,
    perspective=0.0005,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    single_cls=True  # If only detecting text
)