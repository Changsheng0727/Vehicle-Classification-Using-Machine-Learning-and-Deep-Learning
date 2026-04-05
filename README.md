# 车辆检测与分类系统

## 系统概述
本系统实现了一个两阶段的车辆识别流程：
1. **车辆检测阶段**：使用YOLOv3检测图像中的车辆位置，消除复杂背景干扰
2. **车辆分类阶段**：使用ImageNet预训练的ResNet50模型进行微调，对检测到的车辆进行分类

## 系统特点
- ✅ 使用YOLOv3准确定位车辆位置
- ✅ 使用ResNet50进行精细分类
- ✅ 消除复杂背景干扰
- ✅ 目标准确率 > 85%
- ✅ 完整的端到端流程
- ✅ 可视化检测和分类结果

## 文件结构
```
Vehicle-Classification-Using-Machine-Learning-and-Deep-Learning-main/
├── vehicle_detection_classification.py  # 核心模块：YOLOv3检测器 + ResNet50分类器
├── run_vehicle_system.py               # 主程序：训练和运行完整系统
├── test_new_image.py                   # 测试新图像
├── yolov3.cfg                          # YOLOv3配置文件
├── yolov3.weights                      # YOLOv3预训练权重
├── Dataset/                            # 训练数据集
│   ├── Bus/
│   ├── Car/
│   ├── motorcycle/
│   └── Truck/
├── test_images/                        # 测试图像目录（自动创建）
└── vehicle_classifier_resnet50.h5      # 训练好的分类模型（运行后生成）
```

## 安装依赖
```bash
pip install tensorflow opencv-python matplotlib scikit-learn seaborn numpy
```

## 快速开始

### 1. 测试YOLOv3检测器
```bash
python run_vehicle_system.py --mode test
```
这将测试YOLOv3是否能正确加载并检测车辆。

### 2. 训练分类模型
```bash
python run_vehicle_system.py --mode train
```
这将：
- 加载数据集（Dataset/目录）
- 使用ResNet50（ImageNet预训练）构建分类模型
- 训练模型并保存为 `vehicle_classifier_resnet50.h5`
- 显示训练历史和评估结果

### 3. 运行完整系统
```bash
python run_vehicle_system.py --mode full
```
这将运行完整的检测+分类流程，包括：
- 测试YOLOv3检测器
- 训练分类模型
- 测试完整系统性能
- 可视化结果

### 4. 测试新图像
1. 将测试图像放入 `test_images/` 目录
2. 运行：
```bash
python test_new_image.py
```

## Notebook导出工具

如果 `ipynb` 导出的 HTML 文件太大，DeepSeek 无法完整读取，请使用本仓库中的导出脚本将 Notebook 导出为更轻量的格式。

### 功能
- 生成去掉输出的 `*.ipynb` 副本
- 导出为 `html`、`markdown` 或 `script`
- 默认会去掉代码输出，减少 `html` 大小

### 使用方法
1. 进入仓库目录：
```bash
cd "D:\mrGuo\我爱的美在这里\Self-Improvement\class\CS50\deeplearning\Vehicle-Classification-Using-Machine-Learning-and-Deep-Learning-main"
```

2. 导出单个 notebook：
```bash
python export_notebooks_clean.py "VEHICLE CLASSIFICATION_FIXED (1).ipynb" --formats html markdown script --output-dir exported_notebooks
```

3. 导出当前目录下所有 notebook：
```bash
python export_notebooks_clean.py . --formats html markdown script --output-dir exported_notebooks
```

4. 只生成去掉输出的 notebook（不导出 HTML）：
```bash
python export_notebooks_clean.py "VEHICLE CLASSIFICATION_FIXED (1).ipynb" --clean-only --output-dir exported_notebooks
```

5. 仅导出 notebook 输出，隐藏代码输入：
```bash
python export_notebooks_clean.py "VEHICLE CLASSIFICATION_FIXED (1).ipynb" --output-only --formats html markdown --output-dir exported_notebooks
```

### 说明
- `html` 导出会默认使用 `--TemplateExporter.exclude_output=True`，这样会省去输出结果和图片数据
- `markdown` 导出也默认去掉输出，使文件更小、更易搜索
- 如果你只需要代码，`script` 导出会生成一个普通的 Python 脚本

### 输出路径
默认导出位置为：
```
exported_notebooks/
```
如果需要修改输出目录，可使用 `--output-dir` 参数。

## 数据集要求
- 图像格式：JPG、JPEG、PNG、BMP
- 目录结构：
  ```
  Dataset/
  ├── Bus/      # 公交车图像
  ├── Car/      # 小汽车图像
  ├── motorcycle/ # 摩托车图像
  └── Truck/    # 卡车图像
  ```
- 建议每个类别至少50张图像以获得良好效果

## 系统架构

### YOLOv3检测器 (`YOLOv3Detector` 类)
- 使用COCO预训练的YOLOv3模型
- 专门检测车辆类别：car, bus, truck, motorcycle
- 应用非极大值抑制(NMS)消除重复检测
- 可提取车辆区域用于分类

### ResNet50分类器 (`VehicleClassifier` 类)
- 使用ImageNet预训练的ResNet50作为基础模型
- 冻结基础层，只训练自定义顶层
- 添加全局平均池化层、全连接层和Dropout
- 使用数据增强提高泛化能力
- 输出4个类别的概率：Bus, Car, motorcycle, Truck

### 训练策略
- 学习率：0.0001
- 优化器：Adam
- 损失函数：分类交叉熵
- 数据增强：旋转、平移、剪切、缩放、水平翻转
- 早停机制：防止过拟合
- 学习率衰减：当验证损失停滞时降低学习率

## 性能优化建议

### 提高准确率 (>85%)
1. **增加训练数据**
   - 每个类别至少100-200张图像
   - 使用数据增强生成更多变体

2. **调整模型参数**
   - 增加全连接层神经元数量
   - 调整Dropout率
   - 尝试不同的学习率

3. **改进数据预处理**
   - 使用YOLOv3提取更精确的车辆区域
   - 对图像进行标准化处理
   - 平衡各个类别的样本数量

4. **模型架构优化**
   - 尝试其他预训练模型（如EfficientNet、VGG16）
   - 使用更复杂的顶层架构
   - 微调更多ResNet50层

### 处理特殊情况
1. **未检测到车辆**
   - 降低YOLOv3置信度阈值
   - 调整NMS阈值
   - 使用整个图像作为备选

2. **分类置信度低**
   - 增加训练数据多样性
   - 使用更复杂的数据增强
   - 调整模型复杂度

## 示例输出

### 训练过程
```
训练ResNet50分类模型
============================================================
Bus: 95 张图像
Car: 99 张图像
motorcycle: 100 张图像
Truck: 89 张图像

总共收集到 383 张图像
训练集: 245 个样本
验证集: 69 个样本
测试集: 69 个样本

测试准确率: 0.8768
测试损失: 0.4321
```

### 检测结果
```
检测到 2 个车辆
车辆 1: car -> Car (检测置信度: 92.34%, 分类置信度: 88.56%)
车辆 2: truck -> Truck (检测置信度: 85.67%, 分类置信度: 91.23%)
```

## 故障排除

### 常见问题
1. **YOLOv3加载失败**
   - 检查 `yolov3.cfg` 和 `yolov3.weights` 文件是否存在
   - 确保OpenCV版本支持DNN模块

2. **TensorFlow内存错误**
   - 减少批量大小（batch_size）
   - 使用GPU版本TensorFlow
   - 关闭其他占用内存的程序

3. **数据集问题**
   - 检查图像文件格式
   - 确保目录结构正确
   - 验证图像是否能正常读取

4. **模型训练不收敛**
   - 降低学习率
   - 增加训练轮数
   - 检查数据标签是否正确

### 调试建议
1. 运行 `python run_vehicle_system.py --mode test` 验证YOLOv3
2. 检查数据集图像数量和格式
3. 查看训练过程中的损失和准确率曲线
4. 测试单个图像确认系统正常工作

## 扩展功能

### 添加新车辆类别
1. 在数据集中添加新类别目录
2. 更新 `VehicleClassifier` 中的 `class_names`
3. 重新训练模型

### 实时视频处理
可以扩展系统处理视频流：
```python
# 伪代码示例
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # 检测和分类车辆
    results = process_frame(frame)
    # 显示结果
    cv2.imshow('Vehicle Detection', frame)
```

### 部署到生产环境
1. 将模型转换为TensorFlow Lite格式
2. 优化推理速度
3. 创建REST API服务
4. 构建Web或移动应用界面

## 许可证
本项目仅供学习和研究使用。

## 作者
车辆检测与分类系统 - 基于YOLOv3和ResNet50

## 更新日志
- v1.0: 初始版本，实现基本检测和分类功能
- v1.1: 添加数据增强和模型保存功能
- v1.2: 完善错误处理和用户界面