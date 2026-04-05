#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆检测与分类系统
使用YOLOv3进行车辆检测 + ResNet50进行车辆分类
目标准确率 > 85%
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import class_weight
import seaborn as sns
import random
import warnings
warnings.filterwarnings('ignore')

# 深度学习相关库
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

print("TensorFlow版本:", tf.__version__)
print("Keras版本:", keras.__version__)


class YOLOv3Detector:
    """YOLOv3车辆检测器"""
    
    def __init__(self, config_path='yolov3.cfg', weights_path='yolov3.weights'):
        """初始化YOLOv3检测器"""
        self.config_path = config_path
        self.weights_path = weights_path
        
        # 车辆相关的COCO类别
        self.vehicle_classes = ['car', 'bus', 'truck', 'motorcycle']
        
        # COCO数据集的所有类别
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # 加载YOLO模型
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # 获取输出层名称
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        print(f"YOLOv3模型加载成功，输出层: {self.output_layers}")
    
    def detect_vehicles(self, image, confidence_threshold=0.5, nms_threshold=0.4):
        """检测图像中的车辆"""
        height, width = image.shape[:2]
        
        # 创建blob并前向传播
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # 解析检测结果
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # 只保留车辆类别且置信度高的检测
                if confidence > confidence_threshold and self.coco_classes[class_id] in self.vehicle_classes:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # 计算边界框坐标
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # 应用非极大值抑制
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        
        # 提取最终的检测结果
        final_boxes = []
        final_confidences = []
        final_class_ids = []
        
        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(boxes[i])
                final_confidences.append(confidences[i])
                final_class_ids.append(class_ids[i])
        
        return final_boxes, final_confidences, final_class_ids
    
    def extract_vehicle_regions(self, image_path, save_dir='Detected_Vehicles', min_size=64):
        """从图像中提取车辆区域并保存"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return []
        
        # 检测车辆
        boxes, confidences, class_ids = self.detect_vehicles(image)
        
        # 提取车辆区域
        vehicle_regions = []
        
        for i, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            x, y, w, h = box
            
            # 确保边界框在图像范围内
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            # 只保留足够大的区域
            if w >= min_size and h >= min_size:
                vehicle_region = image[y:y+h, x:x+w]
                vehicle_regions.append(vehicle_region)
                
                # 保存提取的车辆区域
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    base_name = os.path.basename(image_path).split('.')[0]
                    save_path = os.path.join(save_dir, f"{base_name}_vehicle_{i}.jpg")
                    cv2.imwrite(save_path, vehicle_region)
        
        return vehicle_regions
    
    def visualize_detection(self, image_path, output_path=None):
        """可视化检测结果"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return 0
        
        boxes, confidences, class_ids = self.detect_vehicles(image)
        
        # 绘制检测结果
        for i, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            x, y, w, h = box
            
            # 绘制边界框
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # 添加标签
            label = f"{self.coco_classes[class_id]}: {confidence:.2f}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 显示或保存结果
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"检测结果已保存到: {output_path}")
        
        # 转换为RGB显示
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title(f"车辆检测结果 - {os.path.basename(image_path)}")
        plt.tight_layout()
        plt.show()
        plt.close()
        
        return len(boxes)


class VehicleClassifier:
    """基于ResNet50的车辆分类器"""
    
    def __init__(self, num_classes=4, input_shape=(224, 224, 3)):
        """初始化分类器"""
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        self.class_names = ['Bus', 'Car', 'motorcycle', 'Truck']
        self.label_map = {name: i for i, name in enumerate(self.class_names)}
        self.reverse_label_map = {i: name for i, name in enumerate(self.class_names)}
    
    def build_model(self):
        """构建基于ResNet50的模型"""
        # 加载预训练的ResNet50模型，不包括顶层
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # 冻结基础模型的前面层
        for layer in base_model.layers:
            layer.trainable = False
        
        # 添加自定义顶层
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        predictions = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # 创建完整模型
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # 编译模型
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("ResNet50模型构建完成")
        self.model.summary()
        
        return self.model
    
    def prepare_data(self, images, labels, test_size=0.2, val_size=0.1, save_processed_dir=None, filenames=None):
        """准备训练、验证和测试数据

        如果指定 save_processed_dir，则会将 resize 后但未归一化的图像保存到该目录。
        """
        # 将图像调整为ResNet50所需的大小
        processed_images = []
        processed_labels = []
        processed_filenames = []

        for idx, img in enumerate(images):
            # 确保图像是3通道
            if len(img.shape) == 2:  # 灰度图像
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:  # RGBA图像
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # 调整大小
            img_resized = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
            processed_images.append(img_resized)
            processed_labels.append(labels[idx])
            processed_filenames.append(filenames[idx] if filenames and idx < len(filenames) else f"img_{idx:05d}.jpg")

        if save_processed_dir:
            self.save_processed_images(processed_images, processed_labels, save_processed_dir, processed_filenames)

        # 转换为numpy数组并进行ResNet50预处理
        X = np.array(processed_images).astype('float32')
        X = preprocess_input(X)
        
        # 将标签转换为one-hot编码
        y = np.array([self.label_map[label] for label in labels])
        y_onehot = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
        
        # 分割数据集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_onehot, test_size=test_size, random_state=42, stratify=y
        )
        
        # 从临时集中分割验证集
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=np.argmax(y_temp, axis=1)
        )
        
        print(f"训练集: {X_train.shape[0]} 个样本")
        print(f"验证集: {X_val.shape[0]} 个样本")
        print(f"测试集: {X_test.shape[0]} 个样本")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_augmentation(self):
        """创建数据增强生成器"""
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        return datagen

    def save_processed_images(self, images, labels, save_dir, filenames=None):
        """将预处理后（已 resize）的图像保存到指定文件夹"""
        for idx, (img, label) in enumerate(zip(images, labels)):
            class_dir = os.path.join(save_dir, label)
            os.makedirs(class_dir, exist_ok=True)
            filename = filenames[idx] if filenames and idx < len(filenames) else f"{label}_{idx:05d}.jpg"
            save_path = os.path.join(class_dir, filename)
            cv2.imwrite(save_path, img)

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """训练模型"""
        # 创建回调函数
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint('best_vehicle_classifier.h5', monitor='val_accuracy', 
                          save_best_only=True, mode='max')
        ]
        
        # 计算训练集类权重，缓解类别不平衡
        y_integers = np.argmax(y_train, axis=1)
        class_weights_array = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_integers),
            y=y_integers
        )
        class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

        # 创建数据增强生成器
        datagen = self.create_data_augmentation()

        # 训练模型
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=max(1, len(X_train) // batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        # 评估模型
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"测试准确率: {test_accuracy:.4f}")
        print(f"测试损失: {test_loss:.4f}")
        
        # 预测
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # 计算分类报告
        print("\n分类报告:")
        print(classification_report(y_true_classes, y_pred_classes, 
                                   target_names=self.class_names))
        
        # 混淆矩阵
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar=False,
            linewidths=0.5,
            linecolor='gray',
            ax=ax
        )
        ax.set_title('混淆矩阵', fontsize=14)
        ax.set_xlabel('预测标签', fontsize=12)
        ax.set_ylabel('真实标签', fontsize=12)
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
        return test_accuracy, y_pred_classes, y_true_classes
    
    def predict_single_image(self, image):
        """预测单张图像"""
        # 预处理图像
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        image_resized = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        image_normalized = image_resized.astype('float32')
        image_preprocessed = preprocess_input(image_normalized)
        image_expanded = np.expand_dims(image_preprocessed, axis=0)
        
        # 预测
        predictions = self.model.predict(image_expanded, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return self.reverse_label_map[predicted_class], confidence, predictions[0]
    
    def save_model(self, path='vehicle_classifier_resnet50.h5'):
        """保存模型"""
        self.model.save(path)
        print(f"模型已保存到: {path}")
    
    def load_model(self, path='vehicle_classifier_resnet50.h5'):
        """加载模型"""
        self.model = tf.keras.models.load_model(path)
        print(f"模型已从 {path} 加载")
        return self.model


def prepare_dataset_with_yolo(image_dir='./Dataset', output_dir='./Processed_Vehicles'):
    """使用YOLOv3准备数据集"""
    categories = ['Bus', 'Car', 'motorcycle', 'Truck']
    
    # 初始化YOLOv3检测器
    detector = YOLOv3Detector()
    
    all_images = []
    all_labels = []
    
    for category in categories:
        category_dir = os.path.join(image_dir, category)
        output_category_dir = os.path.join(output_dir, category)
        os.makedirs(output_category_dir, exist_ok=True)
        
        print(f"处理类别: {category}")
        
        for filename in os.listdir(category_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(category_dir, filename)
                
                # 提取车辆区域
                vehicle_regions = detector.extract_vehicle_regions(
                    image_path, 
                    save_dir=output_category_dir,
                    min_size=64
                )
                
                # 如果没有检测到车辆，使用整个图像
                if len(vehicle_regions) == 0:
                    image = cv2.imread(image_path)
                   