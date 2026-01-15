# Semi-Supervised Chromosome Classification

Dự án phân loại nhiễm sắc thể sử dụng mô hình học máy bán giám sát (Semi-Supervised Learning) để tự động phân loại các nhiễm sắc thể theo 23 loại (1-22, X, Y).

## Tổng quan

Dự án này xây dựng một pipeline hoàn chỉnh cho:
1. **Trích xuất đặc trưng**: Blob features (morphological) + PCA từ ảnh nhiễm sắc thể
2. **Huấn luyện bán giám sát**: Self-training với pseudo-labeling
3. **Đánh giá**: Metrics và visualizations
4. **Inference**: Phân loại tự động cho ảnh chưa gán nhãn

## Cấu trúc Project

```
Final_ComputerVision/
├── data/
│   ├── labeled/              # D_L: Ảnh đã gán nhãn (1-22, X, Y)
│   └── unlabeled/            # D_U: Ảnh chưa gán nhãn
├── src/
│   ├── data/                 # Data loading và splitting
│   ├── features/             # Feature extraction (blob, PCA, augmentation)
│   ├── models/               # Model definitions (MLP)
│   ├── training/             # Training logic (supervised, semi-supervised)
│   ├── evaluation/           # Metrics, calibration, visualization
│   ├── inference/            # Inference pipeline
│   └── utils/                # Logging, model utilities
├── scripts/
│   ├── train.py              # Main training script
│   ├── evaluate.py           # Evaluation script
│   └── infer.py              # Inference script
├── models/                   # Saved models
├── results/                  # Results và visualizations
├── notebooks/                # Jupyter notebooks
└── archive/                  # Code cũ từ đề giữa kỳ
```

## Yêu cầu

### Dependencies

```bash
pip install -r requirements.txt
```

Các thư viện chính:
- `torch` - PyTorch cho neural networks
- `numpy`, `scipy` - Tính toán số học
- `scikit-learn` - PCA, metrics
- `scikit-image` - Image processing
- `opencv-python` - Image I/O và processing
- `matplotlib`, `seaborn` - Visualizations
- `pandas` - Data handling

## Cách sử dụng

### 1. Chuẩn bị dữ liệu

Đảm bảo dữ liệu được tổ chức như sau:
- `data/labeled/`: Chứa các thư mục `1/`, `2/`, ..., `22/`, `X/`, `Y/` với ảnh nhiễm sắc thể đã cắt
- `data/unlabeled/`: Chứa ảnh nhiễm sắc thể chưa gán nhãn (`.jpg`, `.png`)

### 2. Training

Chạy pipeline training hoàn chỉnh (supervised + semi-supervised):

```bash
python scripts/train.py
```

Pipeline sẽ:
1. Load và split dữ liệu (70/15/15 train/val/test)
2. Trích xuất features (blob + PCA)
3. Train supervised baseline
4. Thực hiện semi-supervised self-training (≥300 epochs)
5. Lưu models và training curves

### 3. Evaluation

Đánh giá models trên test set:

```bash
python scripts/evaluate.py
```

Script sẽ:
- Load trained models
- Evaluate trên test set
- Generate confusion matrices, per-class metrics
- So sánh supervised vs semi-supervised

### 4. Inference

Phân loại ảnh chưa gán nhãn:

```bash
python scripts/infer.py
```

Script sẽ:
- Load model và PCA
- Predict cho tất cả ảnh trong `data/unlabeled/`
- Lưu kết quả vào `results/predictions.csv`

## Cấu hình

Chỉnh sửa `src/config.py` để thay đổi:
- Data paths
- Model architecture (hidden_dims)
- Training parameters (learning_rate, batch_size, epochs)
- Semi-supervised parameters (confidence_thresholds, top_k_per_class)
- Feature options (extended features, texture, histogram)

## Pipeline chi tiết

### Feature Extraction

1. **Blob Features**: Morphological properties (Area, Perimeter, Aspect Ratio, Compactness, Eccentricity, Hu moments, etc.)
2. **Texture Features**: LBP (Local Binary Pattern), GLCM (Gray-Level Co-occurrence Matrix)
3. **Histogram Features**: Mean, Std, Skewness, Kurtosis, Percentiles
4. **PCA**: Dimensionality reduction trên flattened image vectors

### Model Architecture

- **MLP**: Simple Multi-Layer Perceptron
  - Input: PCA features + scaled blob features (concatenated)
  - Hidden layers: [128, 64] (có thể tùy chỉnh)
  - Output: 23 classes (softmax)
  - Regularization: Dropout (0.3), Weight decay (L2)

### Semi-Supervised Learning

**Self-Training với Pseudo-Labeling**:
1. Train supervised baseline trên D_L
2. Predict trên D_U với confidence scores
3. Select high-confidence samples (threshold T_conf)
4. Apply top-K per class để tránh class imbalance
5. Add pseudo-labels vào training set
6. Retrain và lặp lại

**Safety mechanisms**:
- Pseudo-labels không bao giờ thay thế labeled data gốc
- Top-K filtering để duy trì class balance
- Early stopping trong mỗi iteration
- Tổng epochs ≥ 300 (yêu cầu đề bài)

## Kết quả

Sau khi chạy training và evaluation, kết quả được lưu trong:
- `models/`: Trained models
- `results/`: Metrics, confusion matrices, training curves
- `results/predictions.csv`: Predictions cho unlabeled data

## Phân công công việc

Project được chia thành các module độc lập để làm việc nhóm:

1. **Data & Features**: `src/data/`, `src/features/`
2. **Models & Training**: `src/models/`, `src/training/`
3. **Evaluation & Reporting**: `src/evaluation/`
4. **Integration**: `scripts/`, `src/inference/`

Xem chi tiết trong plan file.

## Notes

- Model mặc định sử dụng **MLP với features + PCA** (theo yêu cầu đề bài)
- Không sử dụng pretrained deep learning models
- Đảm bảo ≥300 epochs total trong quá trình training
- PCA được fit chỉ trên training set để tránh data leakage

## License

Educational project for Computer Vision course.
