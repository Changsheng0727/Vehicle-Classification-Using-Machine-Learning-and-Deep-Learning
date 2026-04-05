#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新图像
使用训练好的模型对新图像进行车辆检测和分类
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from vehicle_detection_classification import YOLOv3Detector, VehicleClassifier

def test_single_image(image_path, detector, classifier):
    """测试单张图像"""
    print(f"\n处理图像: {os.path.basename(image_path)}")
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像 {image_path}")
        return
    
    # 检测车辆
    boxes, confidences, class_ids = detector.detect_vehicles(image)
    
    if len(boxes) == 0:
        print("未检测到车辆，使用整个图像进行分类")
        vehicle_type, confidence, probs = classifier.predict_single_image(image)
        print(f"分类结果: {vehicle_type} (置信度: {confidence:.2%})")
        
        # 显示概率分布
        print("概率分布:")
        for i, (cls_name, prob) in enumerate(zip(classifier.class_names, probs)):
            print(f"  {cls_name}: {prob:.2%}")
        
        # 显示图像
        plt.figure(figsize=(8, 6))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title(f"分类结果: {vehicle_type} ({confidence:.2%})")
        plt.show()
        
        return vehicle_type, confidence
    
    else:
        print(f"检测到 {len(boxes)} 个车辆")
        
        # 准备结果图像
        result_image = image.copy()
        
        for i, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            x, y, w, h = box
            
            # 提取车辆区域
            vehicle_region = image[y:y+h, x:x+w]
            
            # 分类车辆类型
            vehicle_type, type_confidence, probs = classifier.predict_single_image(vehicle_region)
            
            print(f"\n车辆 {i+1}:")
            print(f"  位置: ({x}, {y}, {w}, {h})")
            print(f"  检测类别: {detector.coco_classes[class_id]}")
            print(f"  检测置信度: {confidence:.2%}")
            print(f"  分类结果: {vehicle_type}")
            print(f"  分类置信度: {type_confidence:.2%}")
            print("  概率分布:")
            for j, (cls_name, prob) in enumerate(zip(classifier.class_names, probs)):
                print(f"    {cls_name}: {prob:.2%}")
            
            # 不同车辆类型使用不同颜色
            colors = {
                'Bus': (255, 0, 0),      # 蓝色
                'Car': (0, 255, 0),      # 绿色
                'motorcycle': (0, 0, 255),  # 红色
                'Truck': (255, 255, 0)   # 青色
            }
            
            color = colors.get(vehicle_type, (0, 255, 0))
            
            # 绘制边界框
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 3)
            
            # 添加标签
            label = f"{vehicle_type}: {type_confidence:.2%}"
            cv2.putText(result_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 显示结果
        result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(result_rgb)
        plt.axis('off')
        plt.title(f"车辆检测与分类结果 - {os.path.basename(image_path)}")
        plt.show()
        
        return vehicle_type, type_confidence


def main():
    """主函数"""
    print("=" * 60)
    print("车辆检测与分类系统 - 新图像测试")
    print("=" * 60)
    
    # 检查必要文件
    if not os.path.exists('yolov3.cfg') or not os.path.exists('yolov3.weights'):
        print("错误: 缺少YOLOv3模型文件")
        print("请确保 yolov3.cfg 和 yolov3.weights 文件存在")
        return
    
    if not os.path.exists('vehicle_classifier_resnet50.h5'):
        print("错误: 分类模型文件不存在")
        print("请先运行训练: python run_vehicle_system.py --mode train")
        return
    
    # 初始化检测器和分类器
    print("初始化检测器和分类器...")
    detector = YOLOv3Detector()
    classifier = VehicleClassifier()
    classifier.load_model('vehicle_classifier_resnet50.h5')
    
    print("✓ 模型加载完成")
    
    # 检查测试图像目录
    test_dir = 'test_images'
    if not os.path.exists(test_dir):
        print(f"\n创建测试目录: {test_dir}")
        os.makedirs(test_dir, exist_ok=True)
        print("请将测试图像放入 test_images/ 目录中")
        return
    
    # 获取测试图像
    test_images = []
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            test_images.append(os.path.join(test_dir, filename))
    
    if len(test_images) == 0:
        print(f"\n{test_dir}/ 目录中没有找到测试图像")
        print("支持的格式: .jpg, .jpeg, .png, .bmp")
        return
    
    print(f"\n找到 {len(test_images)} 张测试图像")
    
    # 测试每张图像
    results = []
    for image_path in test_images:
        result = test_single_image(image_path, detector, classifier)
        if result:
            results.append((os.path.basename(image_path), result))
    
    # 显示总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for filename, (vehicle_type, confidence) in results:
        print(f"{filename}: {vehicle_type} (置信度: {confidence:.2%})")
    
    print(f"\n总共测试了 {len(results)} 张图像")


if __name__ == "__main__":
    main()