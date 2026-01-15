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
│   ├── 1/                    # D_L: Ảnh nhiễm sắc thể loại 1 (đã gán nhãn)
│   ├── 2/                    # D_L: Ảnh nhiễm sắc thể loại 2
│   ├── ...                   # D_L: Các loại 3-22
│   ├── 22/                   # D_L: Ảnh nhiễm sắc thể loại 22
│   ├── X/                    # D_L: Ảnh nhiễm sắc thể X
│   ├── Y/                    # D_L: Ảnh nhiễm sắc thể Y
│   ├── unlabeled/            # D_U: Ảnh nhiễm sắc thể chưa gán nhãn (từ metaphase spread)
│   └── json/                 # Metadata (không dùng trong training)
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
├── models/                   # Saved models (PCA, supervised, semi-supervised)
├── results/                  # Results và visualizations
├── topic.md                  # Yêu cầu chi tiết của đề bài
└── requirements.txt          # Python dependencies
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

**D_L (Labeled Data - Dữ liệu đã gán nhãn):**
- Các folder `data/1/`, `data/2/`, ..., `data/22/`, `data/X/`, `data/Y/` chứa ảnh nhiễm sắc thể đã được cắt và gán nhãn tự động từ Bài tập Giữa kỳ
- Mỗi folder chứa các file ảnh (`.jpg`, `.png`) của nhiễm sắc thể tương ứng

**D_U (Unlabeled Data - Dữ liệu chưa gán nhãn):**
- Folder `data/unlabeled/` chứa tập hợp lớn các ảnh nhiễm sắc thể được trích xuất từ các quang cảnh tế bào chất (metaphase spread) chưa được sắp xếp
- Nếu folder `data/unlabeled/` chưa tồn tại hoặc rỗng, script sẽ tự động tạo unlabeled data từ một phần training set để demo (không khuyến nghị cho production)

**Lưu ý:** Theo yêu cầu đề bài (`topic.md`), D_U phải là tập hợp lớn hơn các ảnh chưa được sắp xếp từ metaphase spread. Nên chuẩn bị folder `data/unlabeled/` với dữ liệu thực tế trước khi training.

### 2. Training

Chạy pipeline training hoàn chỉnh (supervised + semi-supervised):

```bash
python scripts/train.py
```

Pipeline sẽ:
1. **Load D_L**: Load ảnh từ các folder `1/`, `2/`, ..., `22/`, `X/`, `Y/` trong `data/`
2. **Split D_L**: Chia thành train/val/test (70/15/15) với stratified sampling
3. **Load D_U**: Load ảnh chưa gán nhãn từ `data/unlabeled/` (nếu có)
4. **Trích xuất features**: Blob features (morphological) + PCA từ flattened image vectors
5. **Supervised baseline**: Train mô hình ban đầu trên D_L (50 epochs)
6. **Semi-supervised self-training**: 
   - Predict trên D_U với confidence scores
   - Chọn high-confidence samples làm pseudo-labels
   - Thêm vào training set và retrain
   - Lặp lại cho đến khi đạt **≥300 epochs tổng cộng** (yêu cầu đề bài)
7. **Lưu kết quả**: Models, training curves, PCA variance plot

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

- **MLP**: Multi-Layer Perceptron (theo yêu cầu đề bài)
  - Input: PCA features (k=128 hoặc giữ ≥95% variance) + scaled blob features (concatenated)
  - Hidden layers: [512, 256, 128] (theo plan trong `topic.md`)
  - Activation: LeakyReLU
  - Output: 23 classes (softmax) - tương ứng với 1-22, X, Y
  - Regularization: Dropout (0.3), Weight decay (1e-4), BatchNorm (optional)

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

## Yêu cầu theo đề bài (topic.md)

Dự án này thực hiện đúng theo yêu cầu trong `topic.md`:

1. **Dữ liệu đầu vào:**
   - D_L: Tập dữ liệu giám sát từ các folder 1-22, X, Y (đã gán nhãn tự động từ Bài tập Giữa kỳ)
   - D_U: Tập hợp lớn hơn các ảnh nhiễm sắc thể chưa được sắp xếp (từ metaphase spread)

2. **Mô hình và huấn luyện:**
   - Mô hình học máy bán giám sát (Semi-Supervised Learning)
   - Phân chia tập dữ liệu: 70% train / 15% val / 15% test
   - **Tối thiểu 300 epochs** (hoặc vòng lặp huấn luyện)
   - Self-training với pseudo-labeling

3. **Trích xuất đặc trưng:**
   - Blob features: Morphological properties (Area, Perimeter, Aspect Ratio, Eccentricity, Hu moments, ...)
   - PCA: Giảm chiều từ flattened image vectors, giữ ≥95% variance hoặc k=128 components
   - Kết hợp: PCA features + morphological features

4. **Kỹ thuật bán giám sát:**
   - Supervised initialization (50 epochs)
   - Self-training loop với confidence threshold (T_conf) annealing từ 0.98 → 0.85
   - Top-K per class để tránh class imbalance
   - Tổng epochs ≥ 300

## Notes

- Model sử dụng **MLP với features + PCA** (theo yêu cầu đề bài, không dùng pretrained deep learning models)
- **Đảm bảo ≥300 epochs total** trong quá trình training (yêu cầu bắt buộc)
- PCA được fit **chỉ trên training set** để tránh data leakage
- Pseudo-labels không bao giờ thay thế labeled data gốc

## License

Educational project for Computer Vision course.
