import os, json, glob, yaml
from pathlib import Path
from typing import List, Dict
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
"""
def compute_classification_accuracy(model:YOLO, ddata_yaml_path:str):
    #a function to compute the top-1 accuracy of a YOLO model
    
    #load datasat info from the yaml file
    with open(ddata_yaml_path, "r") as f:
        data = yaml.safe_load(f)
        
    test_img_dir = data.get("test")
    #predict test iamges
    results = model.predict(source=test_img_dir,save=False, conf=0.3, iou=0.5)
    
    total_detections = 0
    correct_top1 = 0
    
    for res in results:
        gt_class_id = get_ground_truth(res.path)
        if res.boxes:
            #find the detection wtih the highest confidence
            top_detection = res.boxes[0]
            predicted_class_id = int(top_detection.cls.cpu().numpy())
            
            if predicted_class_id == gt_class_id:
                correct_top1 += 1
            total_detections += 1
    def get_ground_truth(img_path:str):
        return 0 
         
    #calculate top-1 accuracy
    top1_accuracy = correct_top1 / total_detections if total_detections > 0 else 0
    return top1_accuracy
"""

def evaluate_model(model:YOLO, data_yaml_path:str):
    
    #evaluates a yolov8 model and computes a full set pf metrics

    known_classes=['apple', 'banana', 'beef', 'blueberries', 'bread', 'butter', 'carrot', 'cheese', 'chicken', 'chicken_breast', 'chocolate', 'corn', 'eggs', 'flour', 'goat_cheese', 'green_beans', 'ground_beef', 'ham', 'heavy_cream', 'lime', 'milk', 'mushrooms', 'onion', 'potato', 'shrimp', 'spinach', 'strawberries', 'sugar', 'sweet_potato', 'tomato']
    known_classes_indices = list(range(len(known_classes)))
    #step1: get standard mAP metrics using ultralytics
    metrics = model.val(data=data_yaml_path, split="test", conf=0.001, iou=0.5,classes= known_classes_indices)
    mAP_50_95 = float(metrics.box.map)
    mAP_50 = float(metrics.box.map50)
    mAP_75 = float(metrics.box.map75)
   
    #step2: compute top-1 accuracy
    
    #step3: compile and return the summary
    summary ={
        "mAP@0.5-0.95": mAP_50_95,
        "mAP@0.5": mAP_50,
        "mAP@0.75": mAP_75
    }
    return summary



if __name__ == "__main__":
    print("a")
    #pretained model weights
    models_weights = '/Users/a1/Downloads/best100.pt'
    #path to data config file
    data_yaml = '/Users/a1/Downloads/HLCV project/project/data/testdata/testsample1/data.yaml'
    #load the yolo model
    model = YOLO(models_weights)
    #evaluate the model
    evaluation_summary = evaluate_model(model, data_yaml_path=data_yaml)
    
    print("\n--- Model Evaluation Summary ---")
    for metric_name, value in evaluation_summary.items():
        print(f"{metric_name:<20}: {value:.4f}")
