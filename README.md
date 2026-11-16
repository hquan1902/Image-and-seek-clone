# Cải tiến Imagine-and-Seek cho Composed Image Retrieval

Repository này chứa mã nguồn dự án Imagine-and-Seek và phần cải tiến của nhóm cho bài tập lớn Xử lý ảnh. Nội dung chính của đề tài là xây dựng một mô hình truy xuất ảnh theo truy vấn kết hợp:

- một ảnh tham chiếu;
- một câu mô tả yêu cầu thay đổi;
- một gallery chứa các ảnh ứng viên.

Mô hình phải đưa ảnh đích lên vị trí cao nhất có thể trong bảng xếp hạng gallery.

Tài liệu gốc của repository được lưu tại [README_ORIGINAL.md](README_ORIGINAL.md). Khi viết báo cáo, nên đọc README này trước, sau đó đọc [FINAL_PIPELINE.md](FINAL_PIPELINE.md) để xem công thức và kết quả chi tiết.

## 1. Mục tiêu và đóng góp

Phương pháp cuối của nhóm sử dụng:

1. OpenCLIP ViT-g/14 để biểu diễn ảnh và văn bản trong cùng không gian đặc trưng 1.024 chiều.
2. TargetPad để giữ tỉ lệ nội dung ảnh trước khi resize.
3. Static fusion làm điểm xuất phát cho truy vấn ảnh–văn bản.
4. Một Residual Adapter đa miền học cách hiệu chỉnh vector truy vấn.
5. Full-gallery loss, hard-negative loss và CIRR group loss để tăng khả năng phân biệt ảnh đích với các ảnh gần giống.
6. Ensemble ba checkpoint từ seed 42, 7 và 123 để tăng tính ổn định.
7. Paired bootstrap 10.000 lần để kiểm tra độ tin cậy của mức cải thiện.

Phần được huấn luyện là `MultiDomainResidualComposer` gồm 336.960 tham số. ViT-g/14 được giữ cố định và dùng để trích xuất đặc trưng trước khi train.

## 2. Sơ đồ hệ thống

```text
Ảnh tham chiếu
      │
      ├─ TargetPad → ViT-g/14 → đặc trưng ảnh fs ─────┐
      │                                               │
Văn bản sửa đổi → CLIP text encoder → đặc trưng ft ──┼─ Static fusion
                                                      │       +
                                                      └─ Residual Adapter
                                                              │
                                                    Query embedding đã chuẩn hóa
                                                              │
                                            Ensemble seed 42, 7 và 123
                                                              │
                                        Cosine similarity với toàn bộ gallery
                                                              │
                                               Xếp hạng ảnh giảm dần
                                                              │
                                      Recall / Group Recall / mAP
```

## 3. Các bộ dữ liệu đã sử dụng

### CIRR

CIRR chứa ảnh đời thực và các truy vấn sửa đổi ảnh. Tập train được dùng để học quan hệ giữa ảnh tham chiếu, câu sửa đổi và ảnh đích. CIRR còn cung cấp nhóm các ảnh gần giống nhau, vì vậy được dùng thêm để huấn luyện `group_cross_entropy`.

Tập validation được dùng để:

- theo dõi R@1 trong quá trình train;
- kiểm tra CIRR Group R@1;
- chọn checkpoint cân bằng;
- đánh giá R@1, R@5, R@10, R@50 và Group R@1, R@2, R@3.

Quy mô sử dụng:

- Train: 28.225 query, gallery 16.939 ảnh.
- Validation: 4.181 query, gallery 2.297 ảnh.

### FashionIQ

FashionIQ được chia thành `dress`, `shirt` và `toptee`. Mỗi truy vấn có hai caption; loader nối chúng bằng từ `and`. Dữ liệu này bổ sung các thay đổi chi tiết về màu sắc, họa tiết, chất liệu và kiểu dáng.

Ảnh train xuất hiện trong validation được loại trước khi huấn luyện để tránh rò rỉ dữ liệu.

| Category | Train query | Train gallery | Validation query | Validation gallery |
|---|---:|---:|---:|---:|
| Dress | 5.985 | 11.452 | 2.017 | 3.817 |
| Shirt | 5.772 | 18.644 | 2.038 | 6.346 |
| Toptee | 5.720 | 15.702 | 1.961 | 5.373 |

Metric đánh giá là R@10 và R@50 cho từng category.

### CIRCO

CIRCO validation được dùng để kiểm tra khả năng chuyển miền của adapter đã học trên CIRR và FashionIQ sang gallery COCO mở rộng.

- Validation: 220 query.
- Gallery: 123.403 ảnh.
- Metric: R@5, R@10, R@25, R@50 và mAP@5, mAP@10, mAP@25, mAP@50.

CIRCO không tham gia cập nhật trọng số hoặc lựa chọn checkpoint.

## 4. Tiền xử lý và tạo cache

Quy trình xử lý ảnh trong `cache_research.py` và `cache_fashioniq_multisplit.py`:

1. Chuyển ảnh sang RGB.
2. Dùng TargetPad với tỉ lệ 1,25 để đệm cạnh ngắn và hạn chế méo hình.
3. Resize bicubic và center crop về `224 × 224`.
4. Chuyển ảnh thành tensor và chuẩn hóa bằng mean/std của CLIP.
5. ViT-g/14 mã hóa ảnh thành vector 1.024 chiều.
6. CLIP text encoder mã hóa caption thành vector 1.024 chiều.
7. Chuẩn hóa L2 các vector.
8. Lưu đặc trưng, query, gallery ID và metadata vào cache `.pt`.

Việc tạo cache giúp mỗi epoch chỉ huấn luyện adapter nhỏ, không phải mã hóa lại toàn bộ ảnh bằng ViT-g/14.

## 5. Kiến trúc mô hình

Với đặc trưng ảnh tham chiếu `fs`, đặc trưng văn bản `ft` và trọng số ảnh `w`, truy vấn nền là:

```text
q_base = w·fs + (1-w)·ft
```

Đầu vào Residual Adapter:

```text
x = [fs; ft; fs⊙ft; fs-ft]
```

Kiến trúc adapter:

```text
LayerNorm(4096)
→ Linear(4096, 64)
→ GELU
→ Dropout
→ Linear(64, 1024)
```

Query cuối:

```text
q = Normalize(q_base + 0.1·ResidualAdapter(x))
```

Lớp cuối của adapter được khởi tạo bằng 0. Vì vậy tại epoch 0, mô hình bắt đầu đúng từ static fusion rồi học phần residual cần thiết.

## 6. Hàm mất mát và huấn luyện

Hàm mục tiêu gồm bốn thành phần:

```text
L = L_gallery
  + 0.5·L_hard-negative
  + 2.0·L_CIRR-group
  + 0.01·L_residual
```

- `full_gallery_cross_entropy`: đưa ảnh đích lên trên toàn bộ gallery của miền.
- `hard_negative_margin_loss`: tập trung vào 32 ảnh sai có điểm cao nhất, margin 0,05.
- `group_cross_entropy`: phân biệt ảnh đích với các ảnh gần giống trong nhóm CIRR.
- `residual penalty`: hạn chế adapter thay đổi quá mạnh không gian đặc trưng CLIP.

Cấu hình chính:

| Tham số | Giá trị |
|---|---:|
| Epoch tối đa | 30 |
| Batch size | 128 |
| CIRR sample mỗi epoch | 18.000 |
| Sample tối đa mỗi miền FashionIQ | 6.000 |
| Learning rate | 5e-5 |
| Weight decay | 1e-3 |
| Temperature | 0,07 |
| Hidden dimension | 64 |
| Dropout | 0,3 |
| Patience | 8 |

Code trộn batch CIRR, dress, shirt và toptee trong mỗi epoch để một adapter học đồng thời bốn miền.

Checkpoint `best.pt` chỉ được cập nhật khi kết quả tổng hợp tốt hơn và vẫn thỏa điều kiện cân bằng trên CIRR, FashionIQ và CIRR Group Recall.

Ba lần train được thực hiện với seed 42, 7 và 123. Khi suy luận:

```text
q_ensemble = Normalize((q_42 + q_7 + q_123) / 3)
```

## 7. Truy xuất ảnh

Mỗi query và ảnh gallery đều được chuẩn hóa L2. Vì vậy tích vô hướng chính là cosine similarity:

```text
score_j = q_ensembleᵀ · g_j = cosine(q_ensemble, g_j)
```

Ảnh tham chiếu được loại khỏi danh sách ứng viên. Các ảnh còn lại được sắp xếp theo `score_j` giảm dần.

## 8. Kết quả cuối

### CIRR validation: static fusion so với adapter ensemble

| Metric | Static fusion | Adapter ensemble | Tăng điểm % |
|---|---:|---:|---:|
| R@1 | 33,58% | 36,47% | +2,89 |
| R@5 | 62,59% | 67,90% | +5,31 |
| R@10 | 74,55% | 80,10% | +5,55 |
| R@50 | 92,11% | 95,50% | +3,39 |
| Group R@1 | 64,43% | 64,77% | +0,34 |
| Group R@2 | 81,82% | 82,97% | +1,15 |
| Group R@3 | 91,20% | 91,75% | +0,55 |

### FashionIQ validation

| Category | Metric | Static fusion | Adapter ensemble | Tăng điểm % |
|---|---|---:|---:|---:|
| Dress | R@10 | 37,13% | 43,08% | +5,95 |
| Dress | R@50 | 57,76% | 64,65% | +6,89 |
| Shirt | R@10 | 37,88% | 46,57% | +8,69 |
| Shirt | R@50 | 54,17% | 65,11% | +10,94 |
| Toptee | R@10 | 44,42% | 51,96% | +7,54 |
| Toptee | R@50 | 63,54% | 72,21% | +8,67 |

### CIRCO validation: GDN/dispersion fusion so với adapter ensemble

| Metric | GDN/dispersion | Adapter ensemble | Tăng điểm % |
|---|---:|---:|---:|
| R@5 | 34,09% | 35,45% | +1,36 |
| R@10 | 42,27% | 49,09% | +6,82 |
| R@25 | 57,27% | 65,00% | +7,73 |
| R@50 | 65,91% | 71,36% | +5,45 |
| mAP@5 | 18,48% | 18,68% | +0,20 |
| mAP@10 | 19,08% | 20,09% | +1,01 |
| mAP@25 | 21,09% | 22,19% | +1,10 |
| mAP@50 | 22,04% | 22,89% | +0,85 |

Paired bootstrap 10.000 lần xác nhận mức tăng CIRR R@1/5/10/50 và sáu metric FashionIQ có khoảng tin cậy 95% không chứa 0. Trên CIRCO, R@10/25/50 có khoảng tin cậy dương.

## 9. File cần đọc để viết báo cáo

| Nội dung báo cáo | File nguồn |
|---|---|
| Cấu trúc và loader dữ liệu | `src/research_data.py` |
| TargetPad và cache đặc trưng | `cache_research.py`, `cache_fashioniq_multisplit.py` |
| Kiến trúc adapter và loss | `src/research_adapter.py` |
| Quy trình train và chọn checkpoint | `train_multidomain.py` |
| Ensemble và cosine ranking | `retrieve_research.py` |
| Công thức Recall, Group Recall, mAP | `src/research_metrics.py` |
| Paired bootstrap | `compare_retrieval.py` |
| Chi tiết phương pháp và công thức | `FINAL_PIPELINE.md` |
| Demo định tính CIRR | `visualizer/` |

## 10. Artifact kết quả

```text
logs/
├── student_multidomain_group2_seed42.log
├── student_multidomain_group2_seed7.log
└── student_multidomain_group2_seed123.log

runs/
├── student_multidomain_group2_seed42/
├── student_multidomain_group2_seed7/
├── student_multidomain_group2_seed123/
├── student_multidomain_group2_ensemble/
├── student_multidomain_hardneg_seed42/     # static CIRR/FashionIQ
└── core_results/                           # GDN CIRCO

reports/core_results/
├── group2_three_seed_summary.json
├── group2_three_seed_summary.png
├── group2_three_seed_learning_curves.png
├── group2_ensemble_cirr_qualitative.png
└── group2_ensemble_vs_*.json               # bootstrap
```

Mỗi thư mục seed có `best.pt`, `config.json`, `history.json` và các file metric. Thư mục ensemble chứa prediction và metric cuối trên CIRR, ba category FashionIQ và CIRCO.

## 11. Cấu trúc báo cáo đề xuất

1. Đặt vấn đề Composed Image Retrieval.
2. Tổng quan Imagine-and-Seek và bài toán cần cải thiện.
3. Dữ liệu CIRR, FashionIQ và CIRCO.
4. Tiền xử lý ảnh bằng TargetPad và ViT-g/14.
5. Static fusion và Residual Adapter đa miền.
6. Full-gallery, hard-negative và group loss.
7. Thiết lập huấn luyện và ensemble ba seed.
8. Metric Recall, Group Recall và mAP.
9. Kết quả định lượng và paired bootstrap.
10. Minh họa định tính bằng web demo.
11. Kết luận.

## 12. Cài đặt môi trường

```powershell
python -m venv .venv-ramf
.\.venv-ramf\Scripts\Activate.ps1
pip install -r requirements-ramf.txt
python check_env.py
```

Checkpoint OpenCLIP sử dụng: `laion/CLIP-ViT-g-14-laion2B-s12B-b42K`.

`data/` và `features/` được giữ cục bộ vì có dung lượng lớn. Người chỉ đọc code, kết quả và viết báo cáo không cần tải hai thư mục này.

## 13. Lệnh train

Chạy ba lần với seed 42, 7 và 123; thay đồng thời `--seed`, tên output và tên log:

```powershell
python train_multidomain.py `
  --feature-dir features `
  --output runs\student_multidomain_group2_seed42 `
  --epochs 30 `
  --batch-size 128 `
  --samples-per-domain 6000 `
  --cirr-samples 18000 `
  --hidden 64 `
  --dropout 0.3 `
  --residual-scale 0.1 `
  --residual-penalty 0.01 `
  --lr 5e-5 `
  --weight-decay 1e-3 `
  --temperature 0.07 `
  --label-smoothing 0 `
  --hard-negative-k 32 `
  --hard-negative-weight 0.5 `
  --hard-negative-margin 0.05 `
  --cirr-group-loss-weight 2 `
  --patience 8 `
  --seed 42 `
  2>&1 | Tee-Object logs\student_multidomain_group2_seed42.log
```

## 14. Chạy demo

Web demo sử dụng CIRR validation và các prediction đã tính sẵn:

```powershell
python visualizer\server.py
```

Demo hiển thị:

- metric của static fusion và adapter ensemble;
- ảnh tham chiếu và câu sửa đổi;
- hai bảng xếp hạng ảnh;
- vị trí ground truth trước và sau cải tiến;
- các trường hợp cải thiện, không đổi và suy giảm.
