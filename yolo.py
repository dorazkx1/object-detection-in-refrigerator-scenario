from ultralytics import YOLO
#load a pretianed small model 
model = YOLO("yolov8s.pt")
#path to data config file
data_yaml_path= '/Users/a1/Downloads/HLCV project/project/data/DatasetYOLO/data.yaml'
#train the model
print("Starting YOLOv8s model training on the custom dataset...")
results = model.train(data=data_yaml_path, epochs=1)
#evaluate model performance on the validation set
print("\nTraining complete. The best model weights are saved to 'best.pt'.")

