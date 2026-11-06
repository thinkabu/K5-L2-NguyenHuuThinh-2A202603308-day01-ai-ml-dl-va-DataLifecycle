# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy: 06/11/2026**

**Runtime Colab:** GPU-T4

**Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics: 8.4.145**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468; `class_name`="cab" ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`="ImageNet-1K"
2. Record này nói về cả bức ảnh trong ảnh. vì record này là kết quả của bài toán image classification, mô hình đang phân loại toàn bộ ảnh, chứ không xác định một vật thể riêng biệt.
3. Danh sách tên lớp (class list) không phải do model tự nghĩ ra, nó được xác định từ taxonomy và bộ dữ liệu huấn luyện, sau đó được liên kết với các checkpoint của mô hình
4. Cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp vì: cần định danh lớp bằng ID, dễ đọc và hiểu lớp bằng tên, xác định được ngữ cảnh và hệ thống phân loại bằng taxonomy.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần phải quy định rõ tiêu chí xác định lớp đại diện (Primary Label) cho toàn bộ ảnh như: đối tượng chính, mức độ nổi bật, ngữ cảnh, trường hợp nhiều đối tượng, trường hợp không rõ, và cần phân biệt các hướng xây dựng dataset như object-centric classification và scene classification
6. Vì sao không được lấy `score` của model làm ground truth? Vì score là kết quả dự đoán của mô hình không phải nhãn thực tế đã được xác minh

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`="bus" ; `score`=0.912558 ; `bbox_xyxy`=[93.17, 187.95, 223.01, 320.91] ; `bbox_width`=129.84 ; `bbox_height`=132.96
2. Giải thích box bằng lời thường: từ góc trên-trái (93.17, 187.95) đến góc dưới-phải (223.01, 320.91), chiều rộng box: 129.84 pixel, chiều cao box: 132.96 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Ngưỡng 0,2 có 17 prediction, ngưỡng 0,6 có 6 prediction,
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải dành thêm thời gian kiểm tra và xác minh các Bounding Box được mô hình phát hiện như kiểm tra false positives, xác minh nhãn(class), kiểm tra bounding box, loại bỏ detection không hợp lệ, kiểm tra box trùng lập,..
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này):Người gán nhãn phải vẽ Bounding Box bao sát toàn bộ phần vật thể nhìn thấy được, không bao gồm vùng nền xung quanh. Đối với vật thể bị che khuất, chỉ vẽ theo phần thực tế quan sát được, không tự suy đoán hoặc mở rộng khung bao quanh phần bị che. Nếu vật thể bị cắt bởi mép ảnh, Bounding Box phải kết thúc tại biên ảnh. Trường hợp không đủ thông tin để xác định vật thể, cần chuyển cho Reviewer kiểm tra.
6. Trên ảnh `kitchen`, chọn cánh tay người ở mép trái ảnh. Tôi chọn Escalation vì chỉ nhìn thấy một phần cơ thể, chưa đủ căn cứ xác định cách gán nhãn person. Cần hỏi Lab Coach để đảm bảo tính nhất quán của dataset.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`="kitchen-011" ; `class_name`="spoon" ; `score`=0.359147 ; số điểm polygon=`polygon_point_count`=37 ; 3–5 điểm đầu của `polygon_xy`=[[529.0,
    66.0
  ],
  [
    529.0,
    72.0
  ],
  [
    528.0,
    73.0
  ],
  [
    528.0,
    74.0
  ],
  [
    527.0,
    75.0]]
2. box chỉ cung cấp hình chữ nhật bao quanh vật thể, còn polygon mô tả đường biên thực tế của vật thể thông qua tập hợp các điểm tọa độ, cung cấp thông tin hình dạng chi tiết hơn box, đặc biệt hữu ích với những vật thể dài, mỏng hoặc có hình dạng không đều.
3. `instance_id` dùng để làm mã định danh của một đối tượng cụ thể (instance) trong ảnh. Nó **không** phải class ID và **không** phải tracking ID — tôi hiểu là không được nhầm lẫn giữa việc xác định loại đối tượng, phân biệt từng đối tượng và theo dõi đối tượng qua thời gian
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo:
   Người gán nhãn phải vẽ đường biên Mask sát theo đường viền thực tế của từng vật thể, chỉ bao gồm những pixel thuộc về vật thể đó, không lấy phần nền xung quanh.

- Không vẽ mask theo hình chữ nhật box.
- Không gộp nhiều vật thể cùng lớp thành một mask nếu chúng là các instance riêng biệt.
- Với vật thể bị che khuất, chỉ vẽ phần nhìn thấy được, không tự suy đoán phần bị che.
- Với các vật thể tiếp xúc hoặc chồng lấn, phải phân biệt đường biên của từng instance nếu có đủ thông tin.
- Nếu không thể xác định chính xác đường biên, cần chuyển Reviewer kiểm tra.

5. Trên ảnh `kitchen`, Tôi chọn hai chiếc chảo treo trên tường ở khu vực phía trên, gần giữa ảnh, vì chúng nằm sát và có phần chồng lấn lên nhau và che khuất. Tôi chọn Escalation để hỏi Lab Coach vì không thể xác định chính xác toàn bộ đường biên của từng vật thể. Không nên tự suy đoán phần bị che khuất, nhằm đảm bảo tính chính xác và nhất quán của dữ liệu gán nhãn.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn)                                                                                                             | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                                                                                                                                                                                                                                                        | Annotator làm gì?                                                                                                                                                                                                          | Reviewer xem gì?                                                                                                                                                                                                                   |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phân loại ảnh         | 1 nhãn duy nhất cho toàn bộ bức ảnh (`class_id`, `class_name` thuộc taxonomy cố định), không có tọa độ không gian.                                        | Trên ảnh `traffic` (`visuals/classification_top5.png` và JSON rank 1 là `cab` score 0.510915): ảnh xuất hiện đồng thời nhiều đối tượng (taxi, xe buýt, người đi bộ, đèn tín hiệu), rất khó quyết định nhãn duy nhất cho toàn ảnh nếu guideline không định rõ quy tắc ưu tiên chủ thể chính.               | Quan sát toàn bộ ảnh, đối chiếu quy tắc guideline để chọn đúng 1 nhãn đại diện (Primary Label) từ taxonomy; gửi escalation khi ảnh có nhiều chủ thể ngang hàng không rõ độ ưu tiên.                                        | Kiểm tra nhãn được gán có đúng taxonomy và guideline ưu tiên chủ thể hay không; đánh giá tính đại diện của nhãn trên toàn ảnh; yêu cầu rework nếu gán sai chủ thể chính.                                                           |
| Phát hiện vật thể     | Mỗi vật thể là một hộp chữ nhật tọa độ pixel `[x_min, y_min, x_max, y_max]` (`bbox_xyxy`) kèm nhãn lớp (`class_name` thuộc taxonomy).                     | Trên ảnh `kitchen` (`visuals/detection_predictions.png` và JSON): hạ threshold xuống 0.20 xuất hiện 17 boxes gây nhiễu/trùng lặp; cánh tay người ở mép trái bị cắt cụt (khó quyết định có gắn nhãn `person` không) và các vật dụng bị che khuất một phần khó xác định ranh giới bao ngoài.                | Dùng công cụ bounding box vẽ khung chữ nhật bao sát phần nhìn thấy được của từng vật thể; gán nhãn lớp chính xác; không đoán phần bị che; gửi escalation cho các ca bị cắt mép ảnh hoặc che khuất nghiêm trọng.            | Kiểm tra độ khít (tightness) của từng bounding box (không quá rộng lọt nền, không cắt xén vật thể); kiểm tra sót vật thể (FN), thừa box (FP), gán sai class; yêu cầu vẽ lại nếu box chưa chuẩn.                                    |
| Instance segmentation | Mỗi cá thể có một `instance_id` riêng, kèm đa giác khép kín gồm các điểm tọa độ pixel `polygon_xy` bám sát đường biên thực tế và nhãn lớp (`class_name`). | Trên ảnh `kitchen` (`visuals/segmentation_prediction.png` và JSON ví dụ `kitchen-011` spoon): hai chiếc chảo treo tường nằm sát chồng lấn nhau và các vật dụng mỏng/nhỏ khiến mask của model bị răng cưa, lem nhem; rất khó xác định chính xác đường ranh giới pixel phân chia giữa hai vật thể tiếp xúc. | Vẽ đường viền polygon bám sát từng pixel nhìn thấy của từng đối tượng cụ thể; phân tách các instance riêng biệt kể cả khi cùng lớp; không gộp mask; gửi escalation khi không thể xác định ranh giới giữa hai vật chồng đè. | Kiểm tra độ chuẩn xác của biên mask (không lem ra nền, không lẹm vào vật khác); kiểm tra việc tách đúng từng instance riêng biệt (không gộp mask); rà soát kỹ các vùng chồng lấn/tiếp xúc phức tạp để yêu cầu rework nếu chưa đạt. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì?
   Chỉ sử dụng ba ảnh COCO công khai được cung cấp sẵn trong notebook, không tải lên hoặc đưa vào bất kỳ ảnh cá nhân, dữ liệu nội bộ, họ tên hay MSSV nào vào output/báo cáo công khai.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   Tôi sẽ dừng ngay, không xử lý hoặc lưu trữ dữ liệu đó, và báo cho Lab Coach/Mentor (giảng viên hướng dẫn) để được chỉ dẫn cách xử lý phù hợp.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
