# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Đinh Tiến Cảnh
**Khóa:** K4 - Track 3B
**Ngày chạy:** 05/10/2026

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.8417 | 0.8583 | +0.0167 |
| Answer Relevancy | 0.6766 | 0.8224 | +0.1459 |
| Context Precision | 0.9250 | 0.9333 | +0.0083 |
| Context Recall | 0.9083 | 0.9333 | +0.0250 |

Production vượt ngưỡng 0.75 và cao hơn Naive Baseline ở cả bốn metric. Mức cải
thiện lớn nhất nằm ở Answer Relevancy (+0.1459), cho thấy Hybrid Search và
Cross-Encoder giúp câu trả lời tập trung hơn vào câu hỏi. Context Precision và
Context Recall cùng đạt 0.9333, xác nhận cơ chế tìm trên child và trả về parent cung
cấp ngữ cảnh vừa chính xác vừa đầy đủ. Faithfulness đạt 0.8583 nhưng vẫn là chỉ số
cần theo dõi ở các câu có điều kiện, phép tính hoặc xung đột phiên bản.

> `ragas_report.json` chỉ lưu điểm tổng hợp và failure summary, không lưu nguyên văn
> answer/context từng câu. Phần “Got” dưới đây vì vậy ghi đúng giới hạn chứng cứ,
> không tự dựng lại câu trả lời đã mất.

## Bottom-5 Failures

### #1 — Bảo hiểm PVI của nhân viên thử việc

- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** Không. Nhân viên thử việc chưa được hưởng PVI, chỉ tham gia bảo hiểm xã hội bắt buộc.
- **Got:** Không được lưu nguyên văn; điểm trung bình 0.5000, metric thấp nhất là `faithfulness`.
- **Câu trả lời có đúng không?** Không thể xác nhận nguyên văn. Faithfulness thấp cho thấy câu trả lời có thể bổ sung quyền lợi hoặc điều kiện không xuất hiện trong context.
- **Context có chứa đáp án không?** Có đầy đủ trong `thu_viec.md`; `bao_hiem_suc_khoe.md` cũng xác định PVI áp dụng cho nhân viên chính thức.
- **Câu hỏi có cần viết lại không?** Không. Đây là câu hỏi yes/no rõ ràng.
- **Error Tree:** Output thiếu trung thực → Context có đáp án → Query rõ → lỗi ở generation hoặc context nhiễu.
- **Root cause:** Các đoạn về PVI nói chung có thể làm loãng điều kiện quan trọng “nhân viên thử việc”.
- **Suggested fix:** M3 ưu tiên chunk chứa đúng đối tượng “thử việc”; M5 không thêm các quyền lợi không có trong đoạn gốc.

### #2 — Phí tạm ứng quá hạn

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày; phí 2%/tháng là 300.000 VNĐ/tháng, tương đương khoảng 50.000 VNĐ nếu tính pro-rata 30 ngày.
- **Got:** Không được lưu nguyên văn; điểm trung bình 0.7919, metric thấp nhất là `faithfulness`.
- **Câu trả lời có đúng không?** Điểm tổng thể đạt yêu cầu nhưng có nguy cơ khẳng định cách tính pro-rata chưa được tài liệu quy định.
- **Context có chứa đáp án không?** `tam_ung.md` chứa hạn 15 ngày và mức phí 2%/tháng, nhưng không nêu rõ tính tròn tháng hay theo ngày.
- **Câu hỏi có cần viết lại không?** Nên làm rõ quy ước tính phí theo ngày hoặc theo tháng.
- **Error Tree:** Output có phép tính suy diễn → Context thiếu công thức pro-rata → Query ngầm giả định cách tính → lỗi dữ liệu/ground truth.
- **Root cause:** Ground truth sử dụng quy ước 30 ngày trong khi tài liệu nguồn không quy định.
- **Suggested fix:** Bổ sung công thức pro-rata vào tài liệu hoặc sửa đáp án chuẩn để phản ánh đúng giới hạn thông tin.

### #3 — Mua laptop 30 triệu

- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Giám đốc phòng ban phê duyệt; cần xác nhận cấu hình từ CNTT và ít nhất 3 báo giá.
- **Got:** Không được lưu nguyên văn; điểm trung bình 0.8161, metric thấp nhất là `faithfulness`.
- **Câu trả lời có đúng không?** Khả năng cao trả lời được ý chính, nhưng có thể thêm quy trình hoặc chức danh không được context chứng minh.
- **Context có chứa đáp án không?** Có trong `mua_sam.md`, nhưng đáp án trải trên nhiều điều khoản: ngưỡng tiền, thiết bị CNTT và yêu cầu báo giá.
- **Câu hỏi có cần viết lại không?** Không bắt buộc; đây là câu multi-hop hợp lệ. Có thể tách thành ba sub-query để kiểm tra từng điều kiện.
- **Error Tree:** Output có claim thừa → Context đúng nhưng phân tán → Query nhiều vế → lỗi tổng hợp bằng chứng.
- **Root cause:** Reranker phải gom ba điều kiện từ các đoạn khác nhau, còn LLM có thể nối thêm bước thủ tục không có trong nguồn.
- **Suggested fix:** M2 tăng diversity theo section/source; M3 bảo đảm top-k bao phủ đủ ba điều kiện trước khi trả lời.

### #4 — Lương thử việc Junior tối đa

- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** 85% × 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** Không được lưu nguyên văn; điểm trung bình 0.8310, metric thấp nhất là `faithfulness`.
- **Câu trả lời có đúng không?** Điểm tổng thể cao; lỗi còn lại có thể đến từ diễn giải thừa ngoài phép tính cần thiết.
- **Context có chứa đáp án không?** Có. Parent của `bang_luong_2024.md` chứa cả trần lương Junior và tỷ lệ thử việc 85%.
- **Câu hỏi có cần viết lại không?** Không.
- **Error Tree:** Output gần đúng nhưng có claim thừa → Context đủ → Query rõ → lỗi ở cách diễn đạt câu trả lời.
- **Root cause:** Mô hình có thể giải thích thêm chính sách điều chỉnh lương không liên quan.
- **Suggested fix:** M5 giữ nguyên các con số và đơn vị; câu trả lời chỉ nên trình bày công thức cùng kết quả cuối.

### #5 — Chu kỳ đổi mật khẩu

- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** 120 ngày theo chính sách v2.0 hiện hành; quy định 90 ngày của v1.0 đã bị thay thế.
- **Got:** Không được lưu nguyên văn; điểm trung bình 0.8481, metric thấp nhất là `context_precision`.
- **Câu trả lời có đúng không?** Điểm tổng thể tốt, nhưng context còn chứa tài liệu phiên bản cũ hoặc đoạn ít liên quan.
- **Context có chứa đáp án không?** Có. `mat_khau_v2.md` nêu 120 ngày; `mat_khau_v1.md` nêu 90 ngày và trạng thái đã thay thế.
- **Câu hỏi có cần viết lại không?** Không; hệ thống phải mặc định ưu tiên quy định hiện hành.
- **Error Tree:** Output đạt yêu cầu → Context có cả cũ và mới → Query rõ → lỗi lọc phiên bản trước rerank.
- **Root cause:** Metadata chưa được sử dụng để loại văn bản hết hiệu lực trước Hybrid Search/RRF.
- **Suggested fix:** M5 trích xuất `version`, `effective_date`, `status`; M2 lọc `ĐÃ THAY THẾ` khi câu hỏi không yêu cầu lịch sử.

## Case Study (cho presentation)

**Question chọn phân tích:** Bao lâu phải đổi mật khẩu một lần?

1. **Output đúng?** Điểm tổng thể 0.8481 cho thấy kết quả khá tốt, nhưng report không lưu nguyên văn để xác minh tuyệt đối.
2. **Context đúng?** Có đáp án đúng 120 ngày, nhưng đồng thời có bản cũ 90 ngày.
3. **Query rewrite OK?** Query rõ; không nên bắt người dùng phải thêm “theo chính sách hiện hành”.
4. **Fix ở bước:** M5 gắn metadata phiên bản/trạng thái và M2 loại tài liệu đã bị thay thế trước khi fusion.

**Nếu có thêm 1 giờ, sẽ optimize:**

- Bổ sung metadata `version`, `effective_date`, `status` và filter văn bản hết hiệu lực.
- Cache enrichment theo hash để giảm thời gian 356.5 giây cho 100 chunks.
- Lưu answer/context cùng điểm từng câu để failure analysis có bằng chứng trực tiếp.
- Chạy RAGAS nhiều lượt và báo cáo trung bình/độ lệch chuẩn vì LLM judge có tính biến thiên.
