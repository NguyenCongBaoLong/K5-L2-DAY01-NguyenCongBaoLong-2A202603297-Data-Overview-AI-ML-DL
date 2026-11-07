# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 11/07/2026

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này nói về cả bức ảnh. Vì đây là mô hình gán nhãn đại diện cho toàn bộ bối cảnh không gian mà không khoanh vùng cụ thể và cũng không xác định tọa độ của từng xe riêng lẻ.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách đi kèm taxonomy ImageNet-1K đã được định nghĩa từ trước.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
class_id để đảm bảo tính nhất quán lập trình định danh lớp trong một taxonomy, class_name có thể bị trùng lặp hoặc định nghĩa khác nhau giữa các bộ dữ liệu. Vậy nên cần có cả 3 vì nếu thiếu 1 trong 3 sẽ dễ gây nhầm lẫn khi đổi mô hình hoặc tích hợp hệ thống khác
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
guideline cần định nghĩa quy tắc ưu tiên khi ảnh chứa nhiều đối tượng hỗn hợp
6. Vì sao không được lấy `score` của model làm ground truth?
score chỉ thể hiện độ tự tin toán học của mô hình đối với dữ liệu huấn luyện cũ và nó không phản ánh chất lượng nhãn chuẩn xác do con người xác định.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912557 ; `bbox_xyxy`=[383.01, 196.44, 492.35, 345.12] ; `bbox_width`=109.34 ; `bbox_height`=148.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
góc trên-trái ở tọa độ pixel(383.01, 196.44) và góc dưới-phải ở tọa độ pixel(492.35, 345.12).Chiều rộng của hộp là 109.34 pixel và chiều cao là 148.68 pixel
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
ngưỡng 0.20 có 17 prediction và ngưỡng 0.60 có 6 prediction 
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
khi hạ ngưỡng, reviewer phải lọc thủ công, xóa bỏ các hộp nhiễu không đạt chất lượng
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
vẽ hộp bounding box bám sát nhất vào rìa ngoài cùng của vật thể, không để thừa quá 2-3 pixel lề trắng và không được cắt lẹm vào bộ phận của vật thể trừ khi bị che khuất
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
vật thể chiếc cốc(cup) bị che khuất. cần phải viết quy tắc: " Nếu vật thể bị che khuất dưới 70%, vẫn phải vẽ bounding box bao phủ toàn bộ phần ước lượng của vật thể đó. Nếu bị che khuất trên 70%, bỏ qua không gán nhãn"
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
pylygon mô tả chi tiết các đường biên phức tạp, cho phép cô lập hoàn toàn hình dạng vật thể khỏi môi trường.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id dung để định danh và phân biệt các thực thể độc lập xuất hiện trong cùng một ảnh tĩnh. 
Nó không phải class id vì class id là mã phân loại chung cho cả lớp đối tượng
Nó không phải tracking ID vì tracking ID đòi hỏi việc liên kết, theo dõi và đồng nhất danh tính của một vật thể duy nhất qua các khung hình liên tiếp trong một chuỗi video thời gian thực
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Khi thực hiện gán nhãn đa giác cho đối tượng person, click các điểm neo liên tục dọc theo ranh giới tiếp xúc trực tiếp giữa quần áo, da người và môi trường nền với khoảng cách giữa các điểm từ 3-5 pixel. Sai số tối đa không vượt quá 2 pixel so với thực tế
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Khu vực các chồng đĩa ăn màu trắng xếp chồng đè lên nhau ở góc trước bên trái của chiều bàn gỗ.
sử dung guideline do tính phổ biến cao và tối ưu hóa hiệu suất dự án.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |Thẻ văn bản đơn lẻ gán cho toàn bộ bức ảnh  | Ảnh traffic chứa rất nhiều loại xe đan xen phức tạp nhưng chỉ được gán duy nhất một nhãn tổng quát là cab  | Đọc kỹ danh mục Taxonomy và áp dụng quy tắc ưu tiên để chọn một nhãn đúng nhất đại diện cho bối cảnh ảnh.  | Đánh giá xem nhãn được annotator lựa chọn có tuân thủ đúng quy tắc ưu tiên phân cấp lớp trong tài liệu hướng dẫn (Guideline) hay không. |
| Phát hiện vật thể | Tập hợp các tọa độ hộp giới hạn [x_min, y_min, x_max, y_max] tính bằng pixel kèm nhãn lớp  |Một số dụng cụ nấu bếp (thìa, dĩa) treo phía sau bị mô hình nhận diện sót hoàn toàn khi nâng ngưỡng lọc tin cậy lên cao.  |Vẽ các hộp giới hạn (Bounding Box) bao sát rìa ngoài cùng của vật thể (tight bounding box) và gán nhãn phân loại tương ứng.  | Kiểm tra kỹ xem có hộp nào bị vẽ thiếu (missed), vẽ lệch/thừa quá nhiều khoảng trống (loose box), hoặc gán nhãn sai lớp đối tượng hay không. |
| Instance segmentation |Danh sách tọa độ các điểm đa giác khép kín [[x1, y1], [x2, y2], ...] bao quanh ranh giới thực thể.  |Vùng mặt bàn gỗ (dining table) bị che khuất rải rác bởi nhiều dụng cụ nấu nướng khiến việc xác định đường biên liên tục rất khó khăn.  | Click chuột tỉ mỉ từng điểm neo nối tiếp nhau bám sát đường biên thực tế (contour) của vật thể để tạo thành một mặt nạ đa giác kín. |Kiểm tra độ mịn và độ chính xác của đường ranh giới đa giác; rà soát xem mask có bị lấn sang vật thể khác hoặc bỏ sót các bộ phận nhô ra hay không.  |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ được phép tải xuống và xử lý các tập tin hình ảnh công khai có sẵn thuộc bộ dữ liệu COCO được cung cấp trực tiếp trong bài thực hành và Tuyệt đối không tự ý tải lên (upload) các hình ảnh cá nhân, thông tin nhận dạng cá nhân (PII - ảnh khuôn mặt, biển số xe thực tế, giấy tờ tùy thân) hoặc dữ liệu thuộc sở hữu nội bộ/bảo mật của cơ quan, doanh nghiệp lên môi trường đám mây công cộng của Google Colab.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cáo cho Giảng viên hoặc Lab Coach
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
