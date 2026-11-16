# Pipeline và kết quả cuối

Tài liệu này mô tả đúng phương pháp tạo ra artifact trong nhánh, không bao gồm các thí nghiệm đã bị loại.

## 1. Đầu vào và đặc trưng ảnh

Mỗi query gồm ảnh tham chiếu \(I_s\) và câu sửa đổi \(T\). Ảnh được TargetPad về tỉ lệ tối đa 1,25, resize bicubic và center crop về \(224\times224\), sau đó chuẩn hóa theo mean/std của CLIP. OpenCLIP ViT-g/14 biến ảnh thành \(f_s\in\mathbb{R}^{1024}\); text encoder biến câu sửa đổi thành \(f_t\in\mathbb{R}^{1024}\). Hai vector đều được chuẩn hóa L2.

Backbone được giữ cố định và đặc trưng được cache trước khi train. Vì vậy thời gian train adapter ngắn không có nghĩa là bỏ qua xử lý ảnh; phần ViT-g/14 nặng đã chạy ở bước cache.

## 2. Static fusion và Residual Adapter

Đối chứng static:

\[
q_{base}=w f_s+(1-w)f_t.
\]

Adapter nhận vector:

\[
x=[f_s;f_t;f_s\odot f_t;f_s-f_t]
\]

và thực hiện:

```text
LayerNorm(4096) → Linear(4096,64) → GELU → Dropout → Linear(64,1024)
```

Query cuối:

\[
q=\operatorname{Normalize}(q_{base}+0.1r_\theta(x)).
\]

Lớp cuối của adapter được khởi tạo bằng 0 nên epoch 0 đúng bằng static baseline. Adapter có 336.960 tham số học được.

## 3. Huấn luyện

Một adapter dùng chung được học từ CIRR, FashionIQ dress, shirt và toptee. Trong mỗi epoch, code lấy tối đa 18.000 query CIRR và 6.000 query cho mỗi category FashionIQ, tạo batch 128 rồi trộn thứ tự các miền.

Hàm mục tiêu gồm:

\[
\mathcal L = \mathcal L_{gallery}
+0.5\mathcal L_{hard}
+2\mathbb{1}_{CIRR}\mathcal L_{group}
+0.01\|r_\theta\|_2^2.
\]

- `full_gallery_cross_entropy`: phân loại ảnh đích trong toàn bộ gallery, nhiệt độ 0,07.
- `hard_negative_margin_loss`: chọn 32 ảnh sai có điểm cao nhất, margin 0,05.
- `group_cross_entropy`: riêng CIRR, phân biệt target trong nhóm ảnh gần giống.
- residual penalty: hạn chế adapter phá vỡ không gian CLIP ban đầu.

Optimizer là AdamW, learning rate \(5\times10^{-5}\), weight decay \(10^{-3}\), gradient clipping 1,0. Checkpoint cân bằng chỉ được lưu khi CIRR không giảm, ít nhất ba trong bốn miền không giảm, CIRR Group R@1 không giảm và composite score tốt hơn.

## 4. Suy luận

Ba checkpoint seed 42, 7, 123 tạo ba query embedding:

\[
q_{ens}=\operatorname{Normalize}\left(\frac{q_{42}+q_7+q_{123}}{3}\right).
\]

Mỗi ảnh gallery \(g_j\) và query đã được chuẩn hóa nên:

\[
s_j=\cos(q_{ens},g_j)=q_{ens}^{T}g_j.
\]

Ảnh tham chiếu bị gán điểm \(-\infty\), sau đó gallery được sắp xếp theo \(s_j\) giảm dần.

Trên CIRCO, adapter không được train bằng CIRCO. Base image weight được suy ra theo độ phân tán điểm ảnh/text trên gallery cho từng query, rồi adapter đã học từ CIRR/FashionIQ được áp dụng để đánh giá chuyển miền.

## 5. Kết quả cuối

### CIRR validation

| Metric | Static | Ensemble | Chênh lệch điểm % |
|---|---:|---:|---:|
| R@1 | 33,58 | 36,47 | +2,89 |
| R@5 | 62,59 | 67,90 | +5,31 |
| R@10 | 74,55 | 80,10 | +5,55 |
| R@50 | 92,11 | 95,50 | +3,39 |
| Group R@1 | 64,43 | 64,77 | +0,34 |
| Group R@2 | 81,82 | 82,97 | +1,15 |
| Group R@3 | 91,20 | 91,75 | +0,55 |

### FashionIQ validation

| Category | Metric | Static | Ensemble | Chênh lệch điểm % |
|---|---|---:|---:|---:|
| Dress | R@10 | 37,13 | 43,08 | +5,95 |
| Dress | R@50 | 57,76 | 64,65 | +6,89 |
| Shirt | R@10 | 37,88 | 46,57 | +8,69 |
| Shirt | R@50 | 54,17 | 65,11 | +10,94 |
| Toptee | R@10 | 44,42 | 51,96 | +7,54 |
| Toptee | R@50 | 63,54 | 72,21 | +8,67 |

### CIRCO validation: zero-shot transfer

| Metric | GDN/dispersion | Ensemble | Chênh lệch điểm % |
|---|---:|---:|---:|
| R@5 | 34,09 | 35,45 | +1,36 |
| R@10 | 42,27 | 49,09 | +6,82 |
| R@25 | 57,27 | 65,00 | +7,73 |
| R@50 | 65,91 | 71,36 | +5,45 |
| mAP@5 | 18,48 | 18,68 | +0,20 |
| mAP@10 | 19,08 | 20,09 | +1,01 |
| mAP@25 | 21,09 | 22,19 | +1,10 |
| mAP@50 | 22,04 | 22,89 | +0,85 |

Paired bootstrap 10.000 lần cho thấy cải thiện CIRR R@1/5/10/50 và toàn bộ sáu metric FashionIQ có khoảng tin cậy 95% không chứa 0. Trên CIRCO, R@10/25/50 có CI dương; R@5 và các mức mAP chưa đủ bằng chứng thống kê để khẳng định cải thiện.

## 6. Phạm vi kết luận

Các đối chứng trong nhánh là static image/text fusion trên CIRR/FashionIQ và GDN/dispersion fusion trên CIRCO, dùng cùng cache/backbone/gallery. Đây không phải bản tái lập đầy đủ pipeline Imagine-and-Seek có sinh ảnh proxy; báo cáo phải dùng đúng tên đối chứng trên.

