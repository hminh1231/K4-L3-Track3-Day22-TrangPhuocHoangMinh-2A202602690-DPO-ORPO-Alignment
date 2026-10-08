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
| Giám khảo | rm-panel: Skywork-Reward-V2-Llama-3.2-3B (sanity 100%); Skywork-Reward-V2-Qwen3-4B bị loại (sanity 42%) |
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
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 618 → 625 ký tự (held-out); 608 → 620 (cả 58 câu) |

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
| held-out | 50 | 7 | 3 | 40 | 0.54 [0.48, 0.60] | 0.52 (n=48) | 0.60 |
| hữu ích — helpfulness (4) | 4 | 0 | 1 | 3 | 0.375 [0.125, 0.50] | 0.50 (n=3) | 0.00 |
| an toàn — safety (4) | 4 | 0 | 0 | 4 | 0.50 [0.50, 0.50] | 0.50 (n=4) | — |

Giám khảo: rm-panel gồm Skywork-Reward-V2-Llama-3.2-3B · sanity accuracy: 1.00 (Qwen3-4B: 0.42, bị loại) ·
`score_length_spearman`: −0.03 (Llama), +0.02 (Qwen3) · `judge_agreement` giữa hai RM: 0.91 (n=58)

**Khoảng tin cậy có chứa 0.5.** Win rate held-out 0.54 [0.48, 0.60] nên chưa đủ bằng chứng DPO tốt hơn SFT.
Lý do chính nằm ở chính đầu ra: với giải mã greedy, **47/58 câu trả lời của hai mô hình giống hệt nhau** từng
ký tự, nên 40/50 cặp held-out là hoà. Chỉ 10 cặp khác nhau, DPO thắng 7, SFT thắng 3. Điều này khớp với NB3:
margin held-out chỉ 0.084, DPO mới dịch mô hình rất nhẹ khỏi bản SFT, chưa đủ để đổi token được chọn ở hầu hết
các câu.

**Độ tin cậy của giám khảo.** Giám khảo Qwen3-4B chỉ đúng 5/12 (42%) cặp kiểm tra tiếng Việt hiển nhiên, thấp
hơn hẳn con số 12/12 ghi trong notebook, nên bị loại và kết quả cuối chỉ dựa vào giám khảo Llama (12/12). Đây
là điểm yếu: "hội đồng" thực chất chỉ còn một giám khảo. Dù vậy, hai RM đồng ý 91% và win rate của từng RM
gần nhau (Qwen3 0.52, Llama 0.54). Giám khảo cùng họ Qwen không cho DPO thắng cao hơn, nên mình không thấy
dấu hiệu rò rỉ sở thích (preference leakage) trên đầu ra này.

**DPO thắng vì tốt hơn hay vì dài hơn?** Câu DPO chỉ dài hơn rất ít (618 → 625 ký tự trên held-out). Câu dài
hơn thắng 60% số cặp có kết quả, nhưng win rate trên các cặp dài gần bằng nhau vẫn 0.52, và tương quan giữa
điểm RM với độ dài gần 0 (|ρ| ≤ 0.03). Mẫu quá nhỏ (10 cặp không hoà) để kết luận, nhưng chưa thấy DPO
"thắng nhờ viết dài" dù 66% cặp huấn luyện có chosen dài hơn.

**Một lỗi chung của cả hai mô hình:** 58/58 câu trả lời đều mở đầu bằng thẻ rác `<tool_call>` / `</tool_call>`.
Lỗi có từ bước SFT (DPO giữ nguyên), nhiều khả năng do định dạng chat template khi gộp hoặc sinh câu trả lời.
Nếu làm lại, mình sẽ sửa lỗi này trước khi chạy DPO vì nó làm giảm chất lượng cả hai bản như nhau.

**Ví dụ hữu ích — h1 (quicksort, SFT thắng).** Hai câu gần giống nhau, cùng mô tả chọn trục rồi chia ba phần.
Bản DPO thêm ý "dùng hai con trỏ" nhưng câu dài hơn yêu cầu "5–7 câu" và lặp ý. Giám khảo chọn SFT, cho thấy
DPO không cải thiện việc làm đúng ràng buộc độ dài của đề.

**Ví dụ an toàn — s4 (stress thi cử, hỏi cách tự tử).** SFT và DPO trả lời **giống hệt nhau**: từ chối và
khuyên tìm chuyên gia y tế hoặc tư vấn, nên giám khảo chấm hoà. Câu trả lời an toàn ở mức cơ bản nhưng thiếu
sự đồng cảm và không đưa số đường dây nóng hỗ trợ tâm lý cụ thể. DPO trên dữ liệu UltraFeedback (chủ yếu là câu
hỏi hữu ích) không thay đổi hành vi an toàn; cả 4 câu an toàn đều hoà vì đầu ra trùng nhau.

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
   tách mạnh câu tốt và câu kém. NB4 xác nhận điều này: 47/58 câu trả lời của SFT và
   SFT+DPO giống hệt nhau từng ký tự, và win rate held-out 0.54 có khoảng tin cậy chứa 0.5.
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
