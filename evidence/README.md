# Evidence — Day 22: LangSmith + Prompt Versioning

**Học viên:** Cao Văn Cường — 2A202602493
**LangSmith project:** `day22-lab` · **Prompt Hub:** `cao-van-cuong-rag-prompt-v1`, `cao-van-cuong-rag-prompt-v2`

## Danh sách evidence

| File | Nội dung |
|---|---|
| `01_langsmith_traces.png` | Danh sách traces trong project `day22-lab` (> 100 traces: 50 `rag-query` + 50 `ab-rag-query` + traces bước 3) |
| `02_prompt_hub.png` | 2 prompt đã push lên Prompt Hub (V1 trên, V2 dưới) |
| `02_ab_routing_log.txt` | Log A/B routing: pull cả 2 prompt từ Hub, nhãn `[prompt-v1]` / `[prompt-v2]` cho từng câu (V1 = 19, V2 = 31) |
| `03_ragas_scores.png` | Ảnh chụp terminal: điểm RAGAS từng phiên bản + bảng so sánh V1 vs V2 (in từ log của lần chạy) |
| `03_ragas_report.json` | Bản sao `data/ragas_report.json` |
| `04_pii_demo_log.txt` | Demo PIIDetector — 6 test case (email, phone, SSN, thẻ tín dụng, nhiều PII, câu sạch) |
| `04_json_demo_log.txt` | Demo JSONFormatter — 5 test case (hợp lệ, markdown fences, nháy đơn, dấu phẩy thừa, sai hoàn toàn → JSON dự phòng) |

> Hai file log bước 4 có cùng nội dung vì script in cả 2 demo trong một lần chạy (theo hướng dẫn trong CHECKPOINTS.md).

## Cấu hình chạy

| Thành phần | Giá trị |
|---|---|
| Provider | Gemini (free tier) |
| LLM sinh câu trả lời | `gemini-3.5-flash-lite`, temperature 0 |
| LLM judge cho RAGAS | `gemini-3.1-flash-lite`, temperature 0 (tách riêng để không vượt quota 500 request/ngày/model) |
| Embeddings | `models/gemini-embedding-001` |
| Retrieval | FAISS, chunk 500 / overlap 50 (107 chunks), top-k = 3 |

## Kết quả RAGAS (50 cặp QA × 2 phiên bản)

| Metric | V1 — ngắn gọn | V2 — có cấu trúc | Chênh lệch |
|---|---|---|---|
| faithfulness | **0.9814** | 0.9809 | +0.0005 |
| answer_relevancy | **0.8199** | 0.8169 | +0.0030 |
| context_recall | **0.9800** | 0.9796 | +0.0004 |
| context_precision | **0.9633** | 0.9626 | +0.0007 |

- Faithfulness ≥ 0.9 ở **cả hai** phiên bản (mục tiêu ≥ 0.8).
- Độ dài câu trả lời: V1 trung bình **55 từ** (median 54), V2 trung bình **94 từ** (median 92); 4/50 câu trả lời V2 dùng định dạng markdown/danh sách, V1 không có câu nào.

## Phân tích: vì sao V1 nhỉnh hơn V2

**1. Hai phiên bản gần như ngang nhau, và đó là kết quả có thể đoán trước.** Cả hai prompt dùng chung retriever (FAISS, k = 3), nên `context_recall` và `context_precision` đo chất lượng *truy xuất*, gần như không phụ thuộc system prompt. Chênh lệch < 0.001 ở hai chỉ số này chủ yếu là nhiễu của judge LLM và do một số sample V2 bị lỗi mạng (xem mục Lưu ý), không phải khác biệt thật.

**2. Faithfulness: câu trả lời ngắn có ít "bề mặt" để bịa.** RAGAS tách câu trả lời thành các mệnh đề rồi kiểm tra từng mệnh đề có được context hỗ trợ không. V2 dài gần gấp đôi (94 vs 55 từ) và được yêu cầu "xác định các facts, viết có tổ chức", nên sinh nhiều mệnh đề hơn, gồm cả câu diễn giải/tổng hợp. Mỗi mệnh đề thêm vào là một cơ hội trượt khỏi context. V1 bị giới hạn 2–4 câu nên chủ yếu trích lại facts trong context. Tuy vậy, cả hai đều có dòng "Không suy đoán ngoài context" / "chỉ dựa trên context", nên khoảng cách rất nhỏ (0.0005).

**3. Answer relevancy: câu trả lời tập trung thì "đoán ngược" ra câu hỏi tốt hơn.** RAGAS sinh câu hỏi từ câu trả lời rồi so độ tương đồng embedding với câu hỏi gốc. Câu trả lời ngắn, đi thẳng vào ý của V1 cho câu hỏi sinh ra sát với câu gốc hơn. Phần mở rộng và giải thích thêm của V2 làm câu hỏi sinh ra bị "loãng". Đây là chỉ số V1 thắng rõ nhất (+0.003).

**4. Tại sao answer_relevancy thấp hơn các chỉ số khác (~0.82)?** Câu hỏi trong bộ QA viết bằng tiếng Anh, còn LLM trả lời bằng tiếng Việt (do system prompt viết bằng tiếng Việt). Câu hỏi sinh ngược từ câu trả lời tiếng Việt nên lệch ngôn ngữ so với câu gốc, làm giảm cosine similarity. Ngoài ra `strictness` được đặt = 1 (xem Lưu ý) nên điểm dao động hơn so với trung bình trên 3 câu hỏi.

**Kết luận:** Với knowledge base dạng facts ngắn như lab này, **V1 (ngắn gọn) là lựa chọn tốt hơn**: điểm cao hơn hoặc bằng ở cả 4 chỉ số, rẻ hơn khoảng 40% token output. V2 chỉ đáng dùng khi người dùng cần câu trả lời giải thích sâu, và khi đó nên đo thêm các chỉ số khác (ví dụ độ đầy đủ) mà RAGAS 4 chỉ số này không phản ánh.

**Hướng cải thiện:** viết system prompt yêu cầu trả lời cùng ngôn ngữ với câu hỏi (tăng `answer_relevancy`); thử k = 4–5 hoặc reranker để tăng `context_recall`/`context_precision` cho các câu hỏi cần nhiều đoạn.

## Lưu ý kỹ thuật

- **Quota Gemini free tier** (100 embed request/phút, 500 request/ngày/model) nên code có thêm:
  cache FAISS index + embeddings ra đĩa và retry khi gặp 429 (`utils/data_loader.py`, `utils/llm_factory.py`);
  cache LLM bằng SQLite và tách model judge qua `GEMINI_EVAL_MODEL` / `GEMINI_EVAL_API_KEY` (`03_ragas_evaluation.py`, `config.py`).
- `answer_relevancy.strictness = 1`: mặc định RAGAS yêu cầu 3 candidates/lần gọi, Gemini flash-lite không hỗ trợ (lỗi 400 *Multiple candidates is not enabled*).
- **V2 có 8/200 job RAGAS lỗi** do mất mạng lúc chạy (5 `TimeoutError`, 3 lỗi DNS `getaddrinfo failed`). Điểm trung bình V2 được tính bằng `np.nanmean` trên các sample còn lại (faithfulness thiếu 2, answer_relevancy thiếu 4, context_recall và context_precision thiếu 1 trên 50 sample). V1 chạy đủ 200/200 job.
