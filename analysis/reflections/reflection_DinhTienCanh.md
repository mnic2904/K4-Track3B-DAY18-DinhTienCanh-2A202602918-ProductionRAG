# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Đinh Tiến Cảnh
**Khóa:** K4 - Track 3B
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Tách câu bằng regex, encode bằng `all-MiniLM-L6-v2`, rồi ngắt khi cosine similarity thấp hơn 0.85. Cách này giữ câu nguyên vẹn nhưng model phải được cache để tránh chi phí nạp lặp. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Tạo parent tối đa 2048 ký tự và child tối đa 256 ký tự, liên kết bằng `parent_id`. Lượt tích hợp tạo 100 child chunks từ 26 tài liệu, so với 57 chunk paragraph của baseline. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Dùng heading Markdown `#`–`###` làm biên section, giữ bảng/list cùng tiêu đề và lưu tên section trong metadata. |
| BM25 + Dense fusion | M2 | `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | BM25 bắt từ khóa/số liệu; BGE-M3 bắt ngữ nghĩa; RRF cộng `1/(60 + rank + 1)` mà không cần chuẩn hóa hai thang điểm. Production đạt Context Precision 0.9333, cao hơn Baseline 0.9250. |
| Vietnamese segmentation | M2 | `segment_vietnamese()` | `underthesea.word_tokenize(..., format="text")` rồi thay `_` bằng khoảng trắng để query “nghỉ phép” khớp token tài liệu. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | `BAAI/bge-reranker-v2-m3` chấm trực tiếp cặp query–document và rút top-20 xuống top-3. Model được cache; lần đầu nạp chậm nhưng các lần sau tái sử dụng. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Chấm Faithfulness, Answer Relevancy, Context Precision và Context Recall. Lượt `main.py` mới nhất đạt 0.8583 / 0.8224 / 0.9333 / 0.9333; cả bốn metric đều vượt 0.75 và đều cao hơn Baseline. |
| Diagnostic tree | M4 | `failure_analysis()` | Tính trung bình mỗi câu, tìm metric thấp nhất và ánh xạ sang nguyên nhân/hướng sửa. Bottom-5 hiện gồm PVI thử việc, phí tạm ứng, laptop 30 triệu, lương Junior và chu kỳ mật khẩu. |
| Contextual embeddings và HyQA | M5 | `_enrich_single_call()` | Một lần gọi `gpt-4o-mini` trả summary, questions, context và metadata; lượt chạy mới enrichment 100 chunks trong 356.5 giây. Fallback cục bộ giữ pipeline hoạt động khi API lỗi. |

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  - `pytest: The term 'pytest' is not recognized...` và `No module named pytest` khi dùng Python hệ thống.
  - `[WinError 10061] No connection could be made because the target machine actively refused it` khi Hugging Face kiểm tra model.
  - `APIConnectionError(Connection error.)` khi RAGAS chạy trong sandbox không có mạng.
  - `AttributeError: '_thread.RLock' object has no attribute '_recursion_count'` từ `multiprocess.resource_tracker` lúc Python thoát.
  - `Warning: You are sending unauthenticated requests to the HF Hub` khi tải metadata/model không có `HF_TOKEN`.
- **Nguyên nhân gốc rễ & Cách debug:**
  - Phát hiện project có `.venv`, chuyển sang `.venv/Scripts/python.exe -m pytest` để dùng đúng dependency.
  - Model đã có trong cache nhưng thư viện vẫn gửi HEAD request; đặt `HF_HUB_OFFLINE=1` và `TRANSFORMERS_OFFLINE=1` để bỏ retry mạng.
  - Lần chạy sandbox làm toàn bộ job RAGAS lỗi kết nối; dừng lượt không hợp lệ và chạy lại với quyền mạng cho OpenAI API.
  - Cache SentenceTransformer/CrossEncoder trong tiến trình để test và pipeline không nạp model lặp lại.
  - Lỗi `resource_tracker` xảy ra sau khi test/pipeline đã hoàn thành và không làm mất report; đây là incompatibility lúc cleanup của dependency, không phải lỗi nghiệp vụ.
  - Cảnh báo HF Hub không ảnh hưởng kết quả vì model vẫn tải thành công; có thể đặt `HF_TOKEN` hoặc dùng cache/offline mode để giảm retry và tăng hạn mức.
- **Kiến thức còn thiếu & Cách khắc phục:**
  - Cần hiểu sâu hơn cách RAGAS phân rã claim tiếng Việt và độ ổn định giữa các lượt chấm; sẽ lưu per-question trace và chạy lặp để đo variance.
  - Cần bổ sung version-aware retrieval, query decomposition và diversity selection cho câu hỏi nhiều nguồn.
  - Cần benchmark latency trên GPU/ONNX; lượt CPU hiện tại không phù hợp mục tiêu rerank dưới 150 ms.

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Trợ lý tra cứu quy định doanh nghiệp

#### 1. Hiện trạng

- **Pipeline hiện tại:** Nạp Markdown/PDF text layer → hierarchical chunking → contextual enrichment → BM25 + BGE-M3/Qdrant → RRF → BGE reranker → LLM → RAGAS.
- **Vấn đề / Bottlenecks:** Hai PDF scan chưa OCR; enrichment tuần tự mất 356.5 giây/100 chunks; toàn pipeline mất 919 giây; chưa lọc tài liệu hết hiệu lực; report chưa lưu answer/context theo từng câu.

#### 2. Kế hoạch cải tiến

1. **Chunking strategy:** Dùng Hierarchical làm mặc định; dùng Structure-aware cho Markdown chính sách để giữ bảng và điều khoản; chỉ dùng Semantic cho báo cáo tự sự.
2. **Search retrieval:** Giữ Hybrid BM25 + Dense + RRF; thêm metadata filter theo phiên bản/ngày hiệu lực và query decomposition cho câu nhiều vế.
3. **Reranking:** Giữ `BAAI/bge-reranker-v2-m3` khi có GPU; thử FlashRank ONNX cho SLA thấp, đồng thời chọn kết quả đa dạng theo nguồn thay vì top-k thuần điểm.
4. **Evaluation:** Dùng bốn metric RAGAS, thêm exact-match cho số liệu/version, citation correctness và latency p50/p95. Lưu đầy đủ trace mỗi câu và so sánh trên cùng một test set cố định.
5. **Enrichment:** Dùng Contextual Prepend và metadata version/status; chỉ dùng HyQA cho chunk ít từ khóa. Cache kết quả theo hash nội dung và chạy batch/async để giảm thời gian, chi phí.

#### 3. Timeline triển khai

- **Tuần 1:** OCR hai PDF scan, chuẩn hóa metadata version/effective date/status, sửa report lưu per-question answer/context/metrics.
- **Tuần 2:** Thêm query decomposition, source diversity, version filter; benchmark CrossEncoder với FlashRank và chạy A/B RAGAS ít nhất ba lượt.
