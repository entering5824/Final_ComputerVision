PHÂN LOẠI NHIỄM SẮC THỂ SỬ DỤNG MÔ HÌNH HỌC MÁY BÁN GIÁM SÁT
Mục tiêu: Xây dựng một mô hình Học máy Bán giám sát (Semi-Supervised Learning Model) để tự động phân loại các nhiễm sắc thể riêng lẻ, gán nhãn chúng theo 23 loại (1-22, X, Y). Mô hình phải sử dụng tập dữ liệu đã được gán nhãn tự động từ Bài tập Giữa kỳ và khả năng tự cải thiện hiệu suất bằng cách sử dụng tập dữ liệu chưa được sắp xếp (unlabeled data).
Yêu cầu đầu vào:
1.	Tập dữ liệu Giám sát (Labeled Data - D_L): Các thư mục chứa ảnh nhiễm sắc thể đã được cắt, trích xuất và gán nhãn tự động từ Bài tập Giữa kỳ (Folder 1, 2, ..., 22, X, Y).
2.	Tập dữ liệu Chưa được Giám sát (Unlabeled Data - D_U): Một tập hợp lớn hơn các ảnh nhiễm sắc thể được trích xuất từ các quang cảnh tế bào chất (metaphase spread) chưa được sắp xếp (không biết nhãn).
Yêu cầu về Mô hình và Huấn luyện:
1.	Mô hình phải là mô hình Học máy Bán giám sát.
2.	Thực hiện phân chia tập dữ liệu theo chuẩn (Huấn luyện, Kiểm định, Kiểm tra).
3.	Quá trình huấn luyện phải đạt tối thiểu 300 epoch (hoặc vòng lặp huấn luyện nếu sử dụng các kỹ thuật Semi-Supervised không dựa trên epoch truyền thống như Graph-based methods).
4.	Mô hình cuối cùng phải có khả năng phân loại tự động tập nhiễm sắc thể chưa được sắp xếp.
________________________________________
CÁC BƯỚC THỰC HIỆN DỰA TRÊN KỸ THUẬT NGUỒN
Bước 1: Chuẩn bị Dữ liệu và Kỹ thuật Đặc trưng (Feature Engineering)
Do ảnh nhiễm sắc thể (NST) là các "blob" có hình dạng phức tạp, việc đại diện chúng bằng vector tính toán là rất quan trọng.
1.	Trích xuất Thuộc tính Blob: Đối với mỗi ảnh NST đã được cắt, tính toán các thuộc tính hình thái học (morphological properties): 
o	Diện tích (Area): Tổng số pixel.
o	Chu vi (Perimeter) và Tỷ lệ (Ratio).
o	Tâm (Centroid) (mặc dù không cần thiết cho phân loại, nó là thuộc tính cơ bản của blob).
2.	Đại diện Dữ liệu Vector: Chuyển đổi mỗi ảnh NST thành một vector dữ liệu có số chiều cao (một chiều cho mỗi pixel).
3.	Giảm Chiều dữ liệu (Dimensionality Reduction): Sử dụng Phân tích Thành phần Chính (PCA) để giảm chiều của vector dữ liệu, tạo ra một không gian con tuyến tính (linear subspace).
o	PCA giúp tìm các vector cơ sở (EigenChromosomes, tương tự như EigenFaces) để xấp xỉ dữ liệu với lỗi bình phương trung bình cực tiểu (MSE). Điều này là cần thiết vì kích thước ảnh lớn dẫn đến không gian có số chiều cao.
Bước 2: Xây dựng Mô hình Học máy Bán giám sát
Mục tiêu là tận dụng lượng lớn dữ liệu chưa được gán nhãn (D_U) để tăng cường hiệu suất của mô hình, vượt qua giới hạn của tập D_L nhỏ đã được tạo ra ở Bài tập Giữa kỳ.
1.	Huấn luyện ban đầu (Supervised Initialization):
o	Huấn luyện một mô hình phân loại cơ bản (ví dụ: mô hình Mạng nơ-ron đơn giản hoặc SVM) sử dụng các vector đặc trưng đã giảm chiều từ Bước 1, chỉ trên tập dữ liệu giám sát D_L.
2.	Áp dụng Học máy Bán giám sát (Self-Training/Pseudo-Labeling):
o	Sử dụng mô hình đã huấn luyện từ bước 1 để dự đoán nhãn cho các mẫu trong tập chưa giám sát D_U.
o	Lựa chọn Nhãn Giả: Chỉ chọn các mẫu mà mô hình dự đoán với độ tự tin cao (high confidence) (ví dụ: xác suất dự đoán > T_{conf}) và gán nhãn giả (pseudo-label) cho chúng.
o	Mở rộng Tập Huấn luyện: Thêm các mẫu có nhãn giả này vào tập D_L để tạo ra một tập huấn luyện mở rộng D'_L.
o	Tái Huấn luyện: Tái huấn luyện mô hình trên tập D'_L. Quá trình này cần lặp lại trong tối thiểu 300 epoch hoặc nhiều vòng lặp tự huấn luyện để đảm bảo mô hình hội tụ và tận dụng tối đa dữ liệu D_U.
Lưu ý: Mặc dù các nguồn tập trung vào xử lý ảnh cổ điển (như Canh lề, Homography, và Hình thái học), việc sử dụng PCA để xây dựng không gian con là một kỹ thuật mạnh mẽ để đại diện ảnh cho việc phân loại theo chuẩn học máy.
Bước 3: Đánh giá và Ứng dụng
1.	Đánh giá: Sử dụng tập kiểm tra (Test set) đã được giữ lại từ D_L để đánh giá hiệu suất của mô hình cuối cùng (tính chính xác, độ nhạy).
2.	Ứng dụng Tự động: Áp dụng mô hình đã huấn luyện để phân loại hoàn toàn tự động các blob NST trích xuất từ một quang cảnh tế bào chất chưa được sắp xếp (đây là mục tiêu cuối cùng).
________________________________________
BÀI NỘP VÀ ĐÁNH GIÁ
Sản phẩm Nộp:
1.	Báo cáo (Formal Report): 
o	Lý thuyết: Giải thích chi tiết về việc sử dụng PCA cho việc giảm chiều dữ liệu ảnh, các thuộc tính blob đã sử dụng, và phương pháp Học máy Bán giám sát đã chọn (ví dụ: Self-Training).
o	Triển khai: Cấu trúc mô hình, chi tiết về việc chia tập dữ liệu, tham số huấn luyện (learning rate, batch size, threshold confidence T_{conf}), và bằng chứng về việc huấn luyện đủ 300 epoch.
o	Kết quả: Bảng so sánh hiệu suất giữa mô hình Giám sát ban đầu và mô hình Bán giám sát cuối cùng.
2.	Source Code: Mã nguồn tự động, bao gồm cả phần trích xuất đặc trưng (PCA) và mô hình học máy.
3.	Dữ liệu: Dữ liệu đã gán nhãn (D_L) và dữ liệu chưa gán nhãn (D_U) đã sử dụng.
Tiêu chí Đánh giá:
•	Tính chính xác và hiệu suất của mô hình cuối cùng trên dữ liệu kiểm tra.
•	Khả năng tự động hóa và độ tin cậy của mô hình (Mô hình phải hoạt động tốt trên các NST trích xuất từ dữ liệu đầu vào mới).
•	Thực hiện đúng yêu cầu về mô hình Semi-Supervised và huấn luyện tối thiểu 300 epoch.
•	Chất lượng của quá trình trích xuất đặc trưng và giảm chiều (PCA).


PHÂN LOẠI NHIỄM SẮC THỂ — KẾ HOẠCH CHI TIẾT (Semi-Supervised / Self-Training)

Dưới đây là phiên bản cụ thể, có thể triển khai từ yêu cầu của bạn — bao gồm cấu trúc dữ liệu, các bước xử lý, chi tiết PCA, thuật toán bán-giám sát (pseudo-labeling / self-training), siêu tham số mẫu, mã khung (pseudocode) và kiểm thử/triển khai. Mình viết theo hướng thực nghiệm để bạn có thể copy — paste vào notebook và chạy thử.

1) Cấu trúc dữ liệu (file layout)

Gợi ý layout dự án:

dataset/
  labeled/                   # D_L
    01/                      # folder cho cặp NST 1
      img_0001.png
      ...
    02/
    ...
    22/
    X/
    Y/
  unlabeled/                 # D_U (nhiều ảnh hơn, không biết nhãn)
    u_0001.png
    u_0002.png
    ...


Chia D_L thành: train / val / test theo tỉ lệ 70% / 15% / 15% với stratify theo folder (giữ tỉ lệ 23 lớp).

2) Tiền xử lý & trích xuất đặc trưng (cụ thể)

Mục tiêu: từ mỗi ảnh blob (cropped NST) tạo vector đặc trưng để đưa vào PCA và model.

2.1. Chuẩn hóa ảnh (per-image)

Resize về cùng kích thước H×W (ví dụ 128×64 hoặc 128×128 — chọn sao cho giữ tỉ lệ và chi tiết chiều dọc).

Histogram equalization / CLAHE nếu cần để cân sáng.

Lưu: ảnh float in [0,1].

2.2. Augmentation (chỉ áp dụng lúc train supervised/ semi-supervised)

Random rotation ±10–20°

Random vertical/horizontal shift ≤ 5%

Random vertical flip (cẩn trọng: nếu có ý nghĩa sinh học về hướng, không flip)

Small elastic warp / scaling (±5%) để tăng độ bền

2.3. Trích xuất thuộc tính blob (morphological features) — bổ sung cho vector ảnh:

Area (px)

Perimeter

Major axis length, Minor axis length (ellipse fit)

Aspect ratio = major/minor

Solidity (area / convex_area)

Extent (area / bbox_area)

Eccentricity

Orientation (angle)

Hu moments (7) hoặc Zernike moments (tùy)
=> Tổng: ~12–20 scalar features

2.4. Vector ảnh

Flatten ảnh resized thành vector chiều H*W (ví dụ 128×64 = 8192 dims).

Normalize (zero mean, unit variance per-dimension) trước PCA.

2.5. Kết hợp

Kết hợp vector ảnh (sau PCA) + morphological features (scale hóa) thành final feature vector cho model.

3) PCA — thông số và thực thi (EigenChromosomes)

Mục tiêu: giảm chiều, tìm không gian con tuyến tính.

Chuẩn: chạy PCA chỉ trên tập train labeled (không leak val/test).

Số thành phần k: chọn sao cho giữ >= 95% phương sai hoặc cố định k = 128 hoặc k = 256 nếu tập lớn. (Ví dụ ban đầu: k=128)

Lưu: ma trận eigenvectors (PCA model) và mean vector để áp cho dữ liệu unlabeled sau này.

Kết quả: mỗi ảnh → vector PCA (k dims). Ghép thêm morphological scalars → final dim k + m (m ≈ 12–20).

Ghi chú: lưu PCA object (pickle) để tái sử dụng.

4) Mô hình cơ bản (Supervised initialization)

Hai lựa chọn thực tế:

A. Baseline: SVM / RandomForest

SVM (RBF) trên vector PCA + morph features — dùng khi D_L nhỏ.

Tuning C, gamma bằng grid search trên val.

B. NN nhẹ (khuyến nghị)

MLP: input dim = k + m → hidden [512 → 256 → 128] → softmax(23).

Activation: LeakyReLU, BatchNorm optional, Dropout 0.3.

Optimizer: Adam, lr=1e-3 (xem bảng hyperparams).

Ưu tiên: nếu bạn có nhiều D_L và muốn tận dụng unlabeled, dùng NN để dễ fine-tune với pseudo-labels.

5) Chiến lược bán-giám sát: Self-Training (Pseudo-Labeling) — chi tiết triển khai

Mục tiêu: sử dụng D_U lớn để mở rộng D_L bằng nhãn giả chất lượng cao.

Thuật toán tổng quát (ý tưởng)

Train mô hình M trên D_L (khoảng N_init_epochs, ví dụ 50 epoch).

Dự đoán lớp và confidence (softmax prob) cho tất cả mẫu trong D_U.

Chọn mẫu có max_prob >= T_conf làm pseudo-labeled.

T_conf có thể là constant (ví dụ 0.95) hoặc annealing: bắt đầu cao (0.99) rồi hạ dần đến 0.85 sau một số vòng.

Ngoài ra: giới hạn số mẫu thêm mỗi vòng (ví dụ tối đa 5k mẫu) để tránh overwhelm.

Đánh nhãn giả cho những mẫu này → thêm vào D_L tạo D'_L (kèm flag là pseudo).

Re-train/tune M trên D'_L (hoặc fine-tune tiếp). Lặp lại toàn bộ quy trình cho đến khi đạt >= 300 epoch tổng cộng (hoặc 300 rounds nếu dùng round-based).

Sau mỗi vòng, đánh giá trên validation set — nếu performance giảm, rollback phần pseudo-labeled mới nhất (hoặc giảm T_conf).

Cải tiến thực tế (practical tricks)

Confidence calibration: dùng temperature scaling để calib softmax trước khi thresholding.

Per-class quota: tránh bias sang lớp lớn — chọn top-k confident per class hoặc cap số mẫu per class.

Consistency filtering: áp augmentation khác nhau lên cùng 1 unlabeled image; chỉ pseudo-label nếu pred ổn định (consistency) across augmentations.

Label smoothing cho pseudo-labels (ví dụ soft label [0.9, rest uniformly]) để giảm ảnh hưởng noise.

Weighting: khi train lại, cho pseudo-labeled samples trọng số nhỏ hơn labeled (ví dụ w_pseudo = 0.5 → 0.8).

Curriculum: bắt đầu với only very-high-confidence samples, mở dần.

Cụ thể về vòng/lặp và epoch

Yêu cầu của bạn: tối thiểu 300 epoch. Thiết kế:

Phase 0: Supervised init: 50 epoch

Phase 1..K: mỗi phase fine-tune 25 epoch + pseudo selection → lặp ~10 lần → tổng ≥ 300.

Hoặc trực tiếp train NN 300 epoch trên D'_L với periodic pseudo-updates every 10 epochs.

Pseudocode (rõ ràng)
# giả lập python pseudocode
M = init_model()
pca = fit_PCA(train_images)

# prepare features
X_L_train, y_L_train = extract_features(train)
X_L_val, y_L_val = extract_features(val)
X_U = extract_features(unlabeled)

total_epochs = 0
T_conf = 0.98

# initial supervised
train(M, X_L_train, y_L_train, epochs=50)
total_epochs += 50

while total_epochs < 300:
    probs = M.predict_proba(X_U)   # shape (n_u, 23)
    maxp = probs.max(axis=1)
    pseudo_mask = maxp >= T_conf
    X_pseudo = X_U[pseudo_mask]
    y_pseudo = probs[pseudo_mask].argmax(axis=1)

    # optional: per-class quota, consistency check, calibration...
    D_augmented = concat((X_L_train, y_L_train), (X_pseudo, y_pseudo, weight=0.7))

    # fine-tune
    train(M, D_augmented, epochs=10, sample_weights=weights)
    total_epochs += 10

    # optionally anneal T_conf, evaluate val set
    T_conf = max(0.85, T_conf - 0.01)
    val_metrics = evaluate(M, X_L_val, y_L_val)
    if val_metrics drop too much: revert latest pseudo additions or increase T_conf

6) Siêu tham số mẫu (gợi ý cụ thể)
Tham số	Giá trị gợi ý
Image size	128×64 hoặc 128×128
PCA components k	128 (hoặc giữ ≥95% variance)
MLP hidden	[512, 256, 128]
Activation	LeakyReLU
Optimizer	Adam
LR (init)	1e-3 (scheduler: ReduceLROnPlateau hoặc Cosine)
Batch size	64 (nếu dữ liệu nhiều, 128)
Supervised init epochs	50
Fine-tune epochs per pseudo-round	10–25
Tổng epochs	≥ 300
T_conf (init)	0.98 → anneal → 0.85
Pseudo sample weight	0.5–0.8
Dropout	0.25–0.35
Weight decay	1e-4
7) Đánh giá (metrics & reporting)

Accuracy (overall)

Macro F1 (quan trọng khi lớp mất cân bằng)

Precision/Recall per class — in table 23 rows

Confusion matrix (heatmap) — để thấy lỗi lẫn giữa các cặp tương đồng (ví dụ 6 vs 7)

Calibration curve / reliability diagram (để kiểm confidence thresholds)

Ablation: so sánh supervised-only vs semi-supervised (bảng epochs tương ứng).

Learning curves: accuracy/val_loss theo epoch; % pseudo-labeled samples theo thời gian.

8) Inference pipeline (triển khai)

Input: metaphase image → segmentation/extraction pipeline (blob detection) → crop blob images.

Preprocess: resize, normalize, apply PCA transform, combine morph features.

Forward pass model → softmax → output class + confidence.

Nếu muốn, áp confidence threshold để báo “uncertain” cho review thủ công.

Lưu kết quả vào CSV: {image_id, blob_id, class, prob, centroid, bbox, area}

9) Deliverables cụ thể để nộp

report.pdf:

Mục lý thuyết (PCA, pseudo-labeling, lý do chọn tham số)

Thiết kế pipeline & preprocessing

Bảng hyperparams & training logs

Kết quả: bảng so sánh, confusion matrix, learning curves

Limitations & future work

code/:

preprocess.py (crop, morph features, PCA train/apply)

train_supervised.py

self_training.py (vòng lặp pseudo-labeling)

inference.py

models/: saved model checkpoints, PCA object

data/: (nếu có thể nộp) or instructions to reproduce

notebook.ipynb demo inference trên 1 metaphase image

10) Các cải tiến nâng cao (nếu muốn mở rộng)

Mean Teacher / Temporal Ensembling: consistency regularization — dùng unlabeled bằng cách ép model và teacher produce same outputs. Thường cho hệ quả tốt hơn pseudo-labeling đơn thuần.

FixMatch-like: dùng weak+strong augmentation + confident predictions as hard pseudo labels.

Contrastive pretraining (SimCLR): pretrain encoder trên D_U không nhãn rồi fine-tune classifier — hiệu quả khi D_U lớn.

Domain adaptation nếu dữ liệu test có domain shift.

11) Rủi ro & cách khắc phục

Nhãn giả nhiễu làm hỏng model → dùng T_conf cao, weighting thấp, rollback khi val giảm.

Class imbalance (1–22, X, Y có tần suất khác nhau) → per-class quota & stratified sampling.

Overfitting vào pseudolabels → label smoothing + augmentation + weight decay.

PCA leak → Không dùng val/test để fit PCA.