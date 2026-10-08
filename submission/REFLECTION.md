# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Trang Phước Hoàng Minh (2A202602690)
**Khoá:** K4 · Track 3
**Tier đã chạy:** T4
**Ngày:** 2026-10-08

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 16 GB |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1.000 mẫu · 1 epoch (LoRA r=16, lr 2e-4) |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 66% |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 (100 bước, batch hiệu dụng 8) |
| Giám khảo | ⟨NB4⟩ |
| Chi phí | 0 đồng tiền API (giám khảo chạy local trên Colab) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | 31 phút 24 giây (100 bước), chưa tính ~5 phút tính log-prob tham chiếu |
| Loss ghi nhận đầu tiên | 0.6937 (≈ log 2 = 0.6931) |
| Train loss trung bình | 0.6758 |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | ≈ 0.09 |
| Độ chính xác reward trên held-out | 0.73 (bước 25: 0.64) |
| Margin trên held-out | 0.084 (chosen +0.377, rejected +0.293) |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | ⟨NB4⟩ |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Cả `rewards/chosen` lẫn `rewards/rejected` đều bắt đầu từ 0 (loss đầu tiên 0.6937 ≈ log 2, xác nhận mô hình
tham chiếu đúng là bản SFT đã gộp) và **cùng tăng** suốt 100 bước. Trên held-out, chosen đi từ +0.074 (bước 25)
lên +0.377 (bước 100), rejected đi từ +0.059 lên +0.293. Như vậy đây không phải hình mẫu sách giáo khoa
"chosen ↑, rejected ↓": rejected không hề bị đẩy xuống. Margin tăng (0.015 → 0.084) chỉ vì chosen tăng
**nhanh hơn** rejected. Cũng không phải dịch chuyển xác suất, vì chosen không giảm; log-prob tuyệt đối của câu
chosen trên held-out còn tăng nhẹ (−389.9 → −386.8).

Cách giải thích của mình: dữ liệu Sailor2 là on-policy, cả hai câu đều do cùng một mô hình sinh ra nên có
chung văn phong. DPO với LoRA kéo mô hình về phía phân phối chung đó, làm xác suất của cả hai câu cùng tăng, và
kéo câu chosen nhiều hơn một chút. Đường held-out đi cùng hướng và còn mượt hơn đường huấn luyện (margin
train dao động mạnh 0.03–0.09 vì mỗi bước chỉ có 8 cặp), nên mô hình không học thuộc. Chẩn đoán tự động
INTENDED khớp về mặt kỹ thuật (margin > 0, chosen > 0), nhưng che mất chi tiết rejected cũng tăng. Margin
0.084 ở β = 0.1 tương đương chênh lệch log-ratio chỉ ~0.84 nat, tức DPO mới thay đổi mô hình rất nhẹ;
độ chính xác held-out 0.73 cho thấy tín hiệu thật nhưng còn yếu.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | | | | | | | |
| hữu ích — helpfulness (4) | | | | | | | |
| an toàn — safety (4) | | | | | | | |

Giám khảo: ______ · sanity accuracy: ______ · `score_length_spearman` (reward model) hoặc độ nhất quán khi đổi chỗ A/B — position consistency (giám khảo API): ______

_Khoảng tin cậy có chứa 0.5 không? Giám khảo có đáng tin trên tiếng Việt không (xem bộ cặp kiểm tra sanity)? DPO thắng vì câu trả lời tốt
hơn hay vì dài hơn? Hai reward model trong hội đồng (`per_judge`) có cho win rate gần nhau không? Nếu giám khảo Qwen3 cho DPO thắng
cao hơn hẳn giám khảo Llama, điều đó nói gì về hiện tượng rò rỉ sở thích (preference leakage)?
Chọn 2 ví dụ cụ thể (1 câu về độ hữu ích, 1 câu về an toàn) và giải thích._

_Trả lời ở đây._

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | | | | |
| 0.1 | | | | |
| 0.5 | | | | |

_Nếu không chạy: viết giả thuyết 3 câu về điều bạn dự đoán sẽ thấy._

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

**Quyết định: giữ β = 0.1 và lr = 5e-6 cho đúng 1 epoch (100 bước) trên 800 cặp.**

1. *Phương án thay thế.* Tăng tốc độ học lên 1e-5 hoặc chạy 2–3 epoch, hoặc hạ β xuống 0.05 để mô hình được
   phép đi xa khỏi bản SFT hơn. Ở chiều ngược lại có thể giữ lr = 5e-7 như giá trị thường dùng khi tinh chỉnh
   toàn bộ mô hình.
2. *Vì sao chọn.* Với LoRA, 5e-7 gần như không làm reward nhúc nhích sau ~100 bước (ghi chú trong NB3), còn
   lr lớn hoặc nhiều epoch trên 800 cặp dễ học thuộc và dễ đẩy mạnh thiên vị độ dài (66% cặp có chosen dài
   hơn). Trên T4 miễn phí, một epoch đã tốn ~31 phút, nên 1 epoch là mức cân bằng giữa tín hiệu và thời gian.
3. *Kết quả.* Lựa chọn này an toàn: held-out đi cùng hướng với tập huấn luyện, không có dấu hiệu học thuộc,
   độ chính xác held-out lên 0.73. Điều bất ngờ là mức thay đổi quá nhỏ: margin chỉ 0.084 và cả chosen lẫn
   rejected cùng tăng, nghĩa là phần lớn bước cập nhật kéo mô hình về văn phong chung của dữ liệu chứ chưa
   tách mạnh câu tốt và câu kém. ⟨NB4: so với win rate⟩
4. *Làm lại.* Mình sẽ chạy β-sweep (0.05 / 0.1 / 0.5) và thử lr 1e-5 hoặc 2 epoch, theo dõi đồng thời margin
   held-out và độ dài câu trả lời, để xem margin lớn hơn có đi kèm win rate cao hơn trên các cặp dài gần bằng
   nhau hay chỉ làm câu trả lời dài ra.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | | | | |
| GSM8K | | | | |
| Global-MMLU-vi | | | | |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

_Trả lời ở đây._

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | | | | |
| RPO | | | | |
| DPO-norm | | | | |
| LD-DPO | | | | |
| ORPO | | | | |

_Biến thể nào thay đổi độ dài nhiều nhất, và vì sao (dựa vào công thức loss)?_

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | _<... / ... (n=...)>_ |
| Sai số chuẩn ≈ √(p(1−p)/n) | _<...>_ |

_Thành phần reward nào tăng trước (đúng định dạng hay đúng đáp án)? Chênh lệch có vượt nhiễu không?_

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

_(Tuỳ chọn, 1–3 câu)_
