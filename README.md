# Final Multi-domain CIR Pipeline

Nhánh này là bản bàn giao tối giản phục vụ đọc code, viết báo cáo và kiểm tra kết quả cuối. Nó chỉ chứa pipeline đã tạo ra phương pháp được chọn; các thử nghiệm LoRA, partial-backbone và các mô hình không được chọn đã được loại khỏi nhánh.

Toàn bộ cấu trúc mã nguồn của dự án Imagine-and-Seek gốc vẫn được giữ nguyên, gồm `AutoGPTQ/`, `CLIP-main/`, `GLIP/`, `Grounded-Segment-Anything/`, `MIGC/`, `layout_utils/`, `predictor/`, `prompt/` và các script gốc. Nội dung README ban đầu được lưu tại [README_ORIGINAL.md](README_ORIGINAL.md). Việc tinh giản chỉ áp dụng cho các file nghiên cứu bổ sung của nhóm: nhánh này giữ pipeline cuối và không giữ các nhánh thử nghiệm LoRA/partial-backbone thất bại.

## Phương pháp cuối

```text
Ảnh tham chiếu ─ TargetPad ─ ViT-g/14 ─┐
                                      ├─ static fusion + residual adapter
Văn bản sửa đổi ─ CLIP text encoder ──┘
                                      ↓
                    ensemble seed 42, 7, 123
                                      ↓
                   cosine similarity với gallery
                                      ↓
                  Recall / Group Recall / mAP
```

OpenCLIP ViT-g/14 được dùng để trích xuất đặc trưng 1.024 chiều và được giữ cố định. Phần được huấn luyện là `MultiDomainResidualComposer` gồm 336.960 tham số.

## Vai trò từng bộ dữ liệu

| Dữ liệu | Dùng để làm gì | Cách sử dụng |
|---|---|---|
| CIRR train | Học truy xuất ảnh đời thực và phân biệt các ảnh gần giống | Huấn luyện bằng full-gallery CE, hard-negative loss và CIRR group loss |
| CIRR validation | Chọn checkpoint và đánh giá trong miền | R@1/5/10/50 và Group R@1/2/3; đối chứng là static fusion `w_image=0.2` |
| FashionIQ train: dress, shirt, toptee | Học thuộc tính màu sắc, kiểu dáng và cấu trúc thời trang | Ba miền dùng chung một adapter; caption đôi được nối bằng `and`; ảnh train trùng validation bị loại |
| FashionIQ validation | Kiểm tra khả năng tổng quát trên từng category | R@10 và R@50; đối chứng static fusion với trọng số riêng từng category |
| CIRCO validation | Kiểm tra chuyển miền zero-shot trên gallery COCO lớn | Không tham gia train/chọn checkpoint; đánh giá R@5/10/25/50 và mAP@5/10/25/50; đối chứng GDN/dispersion fusion |

Số mẫu thực tế:

| Miền | Train query / gallery | Validation query / gallery |
|---|---:|---:|
| CIRR | 28.225 / 16.939 | 4.181 / 2.297 |
| FashionIQ dress | 5.985 / 11.452 | 2.017 / 3.817 |
| FashionIQ shirt | 5.772 / 18.644 | 2.038 / 6.346 |
| FashionIQ toptee | 5.720 / 15.702 | 1.961 / 5.373 |
| CIRCO | Không huấn luyện | 220 / 123.403 |

## Xử lý dữ liệu

1. `src/research_data.py` đọc annotation, gallery, ảnh tham chiếu, ảnh đích và văn bản sửa đổi.
2. `cache_research.py` và `cache_fashioniq_multisplit.py` áp dụng TargetPad tỉ lệ 1,25, resize/crop 224, chuẩn hóa CLIP, mã hóa ảnh và text bằng ViT-g/14, chuẩn hóa L2 rồi lưu cache `.pt`.
3. `train_multidomain.py` đọc cache thay vì giải mã ảnh trong từng epoch; trộn batch CIRR và ba category FashionIQ để học một adapter chung.
4. `retrieve_research.py` lấy trung bình ba query embedding từ checkpoint seed 42/7/123, tính cosine similarity với toàn bộ gallery, loại ảnh tham chiếu và xếp hạng giảm dần.
5. `src/research_metrics.py` tính metric; `compare_retrieval.py` thực hiện paired bootstrap 10.000 lần.

Chi tiết kiến trúc, loss, lựa chọn checkpoint và kết quả nằm trong [FINAL_PIPELINE.md](FINAL_PIPELINE.md).

## File cần đọc theo thứ tự

1. `src/research_data.py` — loader cho CIRR, FashionIQ và CIRCO.
2. `cache_research.py`, `cache_fashioniq_multisplit.py` — tiền xử lý và tạo cache.
3. `src/research_adapter.py` — adapter và các hàm loss.
4. `train_multidomain.py` — thuật toán train, validation, early stopping và checkpoint.
5. `retrieve_research.py` — static/GDN baseline, checkpoint ensemble và xếp hạng cosine.
6. `src/research_metrics.py` — công thức metric.
7. `compare_retrieval.py` — bootstrap confidence interval.
8. `visualizer/` — demo CIRR đọc prediction đã tính sẵn.

## Artifact đã kèm theo

- `runs/student_multidomain_group2_seed{42,7,123}/best.pt`: ba checkpoint cuối.
- `runs/student_multidomain_group2_seed{42,7,123}/{config,history}.json`: cấu hình và lịch sử epoch.
- `runs/student_multidomain_group2_ensemble/`: prediction và metric ensemble trên cả năm validation split.
- `runs/student_multidomain_hardneg_seed42/*_static*`: đối chứng static CIRR/FashionIQ.
- `runs/core_results/circo_val_g14_targetpad_dispersion*`: đối chứng GDN CIRCO.
- `logs/student_multidomain_group2_seed*.log`: log terminal ba seed.
- `reports/core_results/group2_ensemble_vs_*.json`: paired bootstrap cuối.
- `reports/core_results/group2_three_seed_*`: bảng và biểu đồ ba seed.

`data/` và `features/` không được commit. Muốn chạy lại cần đặt dữ liệu theo cấu trúc mà `src/research_data.py` quy định và tự tạo cache.

## Cài đặt

```powershell
python -m venv .venv-ramf
.\.venv-ramf\Scripts\Activate.ps1
pip install -r requirements-ramf.txt
python check_env.py
```

Checkpoint OpenCLIP đã dùng là `laion/CLIP-ViT-g-14-laion2B-s12B-b42K`. Khi tạo cache trên máy khác, truyền đường dẫn file `open_clip_pytorch_model.bin` đã tải về vào `--pretrained`.

## Lệnh train cuối

Chạy cùng cấu hình dưới đây ba lần, chỉ thay `--seed` và tên output/log thành `42`, `7`, `123`:

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

## Ví dụ tạo kết quả ensemble CIRR

```powershell
python retrieve_research.py `
  --cache features\cirr_val_g14_targetpad.pt `
  --ensemble-checkpoints `
    runs\student_multidomain_group2_seed42\best.pt `
    runs\student_multidomain_group2_seed7\best.pt `
    runs\student_multidomain_group2_seed123\best.pt `
  --output runs\student_multidomain_group2_ensemble\cirr_val.json
```

## Chạy demo

Demo cần ảnh CIRR validation trong `data/CIRR`:

```powershell
python visualizer\server.py
```

Sau đó mở địa chỉ được in trong terminal. Demo so sánh static fusion và adapter ensemble trên đúng cùng query/gallery CIRR validation.

