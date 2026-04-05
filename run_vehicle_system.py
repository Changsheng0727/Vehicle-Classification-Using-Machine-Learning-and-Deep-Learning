#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆检测与分类系统主程序
运行完整流程：YOLOv3检测 + ResNet50分类
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from vehicle_detection_classification import YOLOv3Detector, VehicleClassifier

def test_yolo_detection():
    """测试YOLOv3检测器"""
    print("=" * 60)
    print("测试YOLOv3车辆检测器")
    print("=" * 60)
    
    # 检查YOLOv3文件
    if not os.path.exists('yolov3.cfg'):
        print("错误: yolov3.cfg 文件不存在")
        return False
    
    if not os.path.exists('yolov3.weights'):
        print("错误: yolov3.weights 文件不存在")
        return False
    
    print(f"yolov3.weights 文件大小: {os.path.getsize('yolov3.weights') / (1024*1024):.2f} MB")
    
    try:
        # 初始化检测器
        detector = YOLOv3Detector()
        print("✓ YOLOv3检测器初始化成功")
        
        # 查找测试图像
        test_images = []
        categories = ['Bus', 'Car', 'motorcycle', 'Truck']
        
        for category in categories:
            category_dir = os.path.join('./Dataset', category)
            if os.path.exists(category_dir):
                images = [f for f in os.listdir(category_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    test_images.append(os.path.join(category_dir, images[0]))
        
        if test_images:
            print(f"找到 {len(test_images)} 张测试图像")
            
            # 测试第一张图像
            test_image = test_images[0]
            print(f"测试图像: {test_image}")
            
            # 可视化检测结果
            num_detections = detector.visualize_detection(test_image)
            print(f"✓ 检测到 {num_detections} 个车辆")
            
            return True
        else:
            print("未找到测试图像")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def prepare_training_data():
    """准备训练数据"""
    print("\n" + "=" * 60)
    print("准备训练数据")
    print("=" * 60)
    
    categories = ['Bus', 'Car', 'motorcycle', 'Truck']
    
    # 收集所有图像和标签
    all_images = []
    all_labels = []
    
    for category in categories:
        category_dir = os.path.join('./Dataset', category)
        if not os.path.exists(category_dir):
            print(f"警告: {category_dir} 目录不存在")
            continue
        
        images = [f for f in os.listdir(category_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"{category}: {len(images)} 张图像")
        
        for filename in images:
            image_path = os.path.join(category_dir, filename)
            image = cv2.imread(image_path)
            if image is not None:
                all_images.append(image)
                all_labels.append(category)
    
    print(f"\n总共收集到 {len(all_images)} 张图像")
    
    # 显示类别分布
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count} 张")
    
    return all_images, all_labels


def train_classification_model():
    """训练分类模型"""
    print("\n" + "=" * 60)
    print("训练ResNet50分类模型")
    print("=" * 60)
    
    # 准备数据
    images, labels = prepare_training_data()
    
    if len(images) == 0:
        print("错误: 没有找到训练图像")
        return None
    
    # 初始化分类器
    classifier = VehicleClassifier()
    classifier.build_model()
    
    # 准备数据集，并保存预处理后的图像以便复查
    X_train, X_val, X_test, y_train, y_val, y_test = classifier.prepare_data(
        images, labels,
        save_processed_dir='./Processed_ResNet50'
    )
    
    # 训练模型
    print("\n开始训练模型...")
    history = classifier.train(X_train, y_train, X_val, y_val, epochs=30, batch_size=16)
    
    # 评估模型
    print("\n评估模型性能...")
    test_accuracy, y_pred, y_true = classifier.evaluate(X_test, y_test)
    
    # 保存模型
    classifier.save_model('vehicle_classifier_resnet50.h5')
    
    # 绘制训练历史
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return classifier, test_accuracy


def run_complete_system():
    """运行完整系统"""
    print("=" * 60)
    print("车辆检测与分类系统")
    print("=" * 60)
    
    # 1. 测试YOLOv3检测器
    if not test_yolo_detection():
        print("YOLOv3检测器测试失败，请检查配置")
        return
    
    # 2. 训练分类模型
    classifier, accuracy = train_classification_model()
    
    if classifier is None:
        print("分类模型训练失败")
        return
    
    print(f"\n✓ 系统训练完成，测试准确率: {accuracy:.2%}")
    
    if accuracy >= 0.85:
        print("✓ 达到目标准确率 (>85%)")
    else:
        print("⚠ 未达到目标准确率，建议：")
        print("  1. 增加训练数据")
        print("  2. 调整模型参数")
        print("  3. 使用更复杂的数据增强")
    
    # 3. 测试完整系统
    print("\n" + "=" * 60)
    print("测试完整系统")
    print("=" * 60)
    
    # 初始化检测器
    detector = YOLOv3Detector()
    
    # 加载分类模型
    classifier.load_model('vehicle_classifier_resnet50.h5')
    
    # 测试一些图像
    test_cases = []
    categories = ['Bus', 'Car', 'motorcycle', 'Truck']
    
    for category in categories:
        category_dir = os.path.join('./Dataset', category)
        if os.path.exists(category_dir):
            images = [f for f in os.listdir(category_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                test_cases.append((os.path.join(category_dir, images[0]), category))
    
    print(f"将测试 {len(test_cases)} 张图像")
    
    for i, (image_path, true_label) in enumerate(test_cases[:3]):  # 只测试前3张
        print(f"\n测试图像 {i+1}: {os.path.basename(image_path)} (真实类别: {true_label})")
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"  无法读取图像")
            continue
        
        # 检测车辆
        boxes, confidences, class_ids = detector.detect_vehicles(image)
        
        if len(boxes) == 0:
            print("  未检测到车辆，使用整个图像进行分类")
            vehicle_type, confidence, _ = classifier.predict_single_image(image)
            print(f"  分类结果: {vehicle_type} (置信度: {confidence:.2%})")
        else:
            print(f"  检测到 {len(boxes)} 个车辆")
            
            for j, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                x, y, w, h = box
                
                # 提取车辆区域
                vehicle_region = image[y:y+h, x:x+w]
                
                # 分类车辆类型
                vehicle_type, type_confidence, _ = classifier.predict_single_image(vehicle_region)
                
                print(f"  车辆 {j+1}: {detector.coco_classes[class_id]} -> {vehicle_type} "
                      f"(检测置信度: {confidence:.2%}, 分类置信度: {type_confidence:.2%})")
        
        # 可视化结果
        result_image = image.copy()
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x, y, w, h = box
            
            # 不同车辆类型使用不同颜色
            colors = {
                'Bus': (255, 0, 0),      # 蓝色
                'Car': (0, 255, 0),      # 绿色
                'motorcycle': (0, 0, 255),  # 红色
                'Truck': (255, 255, 0)   # 青色
            }
            
            # 提取车辆区域并分类
            vehicle_region = image[y:y+h, x:x+w]
            vehicle_type, type_confidence, _ = classifier.predict_single_image(vehicle_region)
            
            color = colors.get(vehicle_type, (0, 255, 0))
            
            # 绘制边界框
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 3)
            
            # 添加标签
            label = f"{vehicle_type}: {type_confidence:.2%}"
            cv2.putText(result_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 显示结果
        result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(result_rgb)
        plt.axis('off')
        plt.title(f"检测与分类结果 - {os.path.basename(image_path)}")
        plt.show()
    
    print("\n" + "=" * 60)
    print("系统运行完成")
    print("=" * 60)
    print("使用方法:")
    print("1. 将新图像放在 test_images/ 目录下")
    print("2. 运行 python test_new_image.py 进行测试")
    print("3. 模型已保存为 vehicle_classifier_resnet50.h5")


def quick_test():
    """快速测试系统"""
    print("快速测试车辆检测与分类系统...")
    
    # 检查必要文件
    if not os.path.exists('yolov3.cfg') or not os.path.exists('yolov3.weights'):
        print("错误: 缺少YOLOv3模型文件")
        print("请确保 yolov3.cfg 和 yolov3.weights 文件存在")
        return
    
    # 检查数据集
    if not os.path.exists('./Dataset'):
        print("错误: Dataset 目录不存在")
        return
    
    # 运行测试
    test_yolo_detection()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='车辆检测与分类系统')
    parser.add_argument('--mode', choices=['test', 'train', 'full'], default='test',
                       help='运行模式: test=测试YOLOv3, train=训练模型, full=完整流程')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        quick_test()
    elif args.mode == 'train':
        train_classification_model()
    elif args.mode == 'full':
        run_complete_system()
    else:
        print("请选择运行模式:")
        print("  python run_vehicle_system.py --mode test   # 测试YOLOv3")
        print("  python run_vehicle_system.py --mode train  # 训练分类模型")
        print("  python run_vehicle_system.py --mode full   # 运行完整系统")