from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

results = model.train(
    data="../utils/config.yaml",
    epochs=10,
    augment=True,
    val=True,
    patience=10
)
