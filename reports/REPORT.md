# Báo cáo Ngày 3 — Tracking Annotation

Chép file này thành `reports/REPORT.md` rồi điền. Giữ nguyên các tiêu đề.

Họ tên / nhóm: `Phan Tấn Đạt`
Ngày: `15/9/2026`

---

## 1. Quá trình gán nhãn

| Mục | Giá trị |
| --- | --- |
| Công cụ | CVAT |
| Thời gian gán `clip_02` (warm-up) | `30` phút |
| Thời gian gán `clip_01` | `90` phút |
| Số track đã vẽ trong `clip_01` | `8` |
| Số keyframe trung bình mỗi track | `79` |

Ba tình huống khó nhất khi gán clip này, và bạn xử lý thế nào:

1. `Xe con vào frame cùng lúc với xe buýt => bắt đầu track xe con khi nhìn thấy rõ đèn pha, bbox to dần khi nhìn thấy rõ thêm bộ phận xe.`
2. `Xe con vào frame cùng lúc với xe buýt nhưng ẩn 1 phần ở sau đuôi xe buýt => bắt đầu track xe con khi nhìn thấy, bbox chỉ bbox những khoảng nhìn thấy thuộc về xe con.`
3. `Xe con đỏ xuất hiện đột ngột ở góc dưới với lượng frame ngắn => check kỹ từng frame đảm bảo không sót xe nào.`

## 2. Tự kiểm và kiểm chéo

Ba lượt tua bắt được gì (lượt 1 nhìn ID, lượt 2 frame đầu/cuối, lượt 3 frame giữa):

- Lượt 1: `đã 'chia để trị' các frame kĩ trong lúc gắn bbox => không có lỗi`
- Lượt 2: `đã 'chia để trị' các frame kĩ trong lúc gắn bbox => không có lỗi`
- Lượt 3: `đã 'chia để trị' các frame kĩ trong lúc gắn bbox => không có lỗi`

Kiểm chéo với: `...`. Chi tiết ở `reports/review_partner.md`.
Số lỗi bạn tìm được trong bản của bạn ấy: `...`. Số lỗi bạn ấy tìm được trong bản của bạn: `...`.

Ca nào hai người quyết khác nhau, và luật nào còn thiếu trong `GUIDELINE_MINI.md`?

`...`

## 3. Pre-gold lock và chấm trước/sau rework

| Evidence | Giá trị |
| --- | --- |
| SHA-256 từ `evidence/pre-gold/clip_01/manifest.json` | `d0e2151b866d365e929b9f7ee512ff840e2f81a619a70d1e8553a20f818e14ed` |
| Thời điểm khóa | `2:08PM 15/9/2026` |
| Số row / frame / track trước khi mở reference | `635 rows, 8 tracks` |

| | HOTA | DetA | AssA | LocA | IDF1 | MOTA | MOTP | FP | FN | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bản pre-gold |0.800 |0.777 |0.827 |0.888 |0.937 |0.867 |0.878 |69 |7 |0 |
| Sau rework |0.800 |0.777 |0.827 |0.888 |0.937 |0.867 |0.878 |69 |7 |0 |
Qua cổng (`IDF1 >= 0.80`, `MOTA >= 0.75`, `MOTP >= 0.70`): **có**

Sau khi đọc danh sách lỗi, bạn đã sửa cụ thể những gì? Ghi theo frame và ID:

| Loại lỗi | Frame | ID | Đã sửa thế nào |
| --- | --- | --- | --- |
|BBOX TRÔI |79 |5 |Sửa lại kích thước box |
|BBOX TRÔI |105 |6 |Sửa lại kích thước box |
|BBOX TRÔI |110 |6 |Sửa lại kích thước box |

## 4. Kết quả model: ByteTrack control vs ReID treatment

Cấu hình từ `outputs/model_run_config.json`:

| Mục | Giá trị |
| --- | --- |
| Python / ultralytics / torch / lap | `3.13.15 /8.4.145 /2.11.0+cu128 /0.5.13` |
| weights / hai tracker | `yolo26n.pt / bytetrack.yaml + /content/Day3-Lab/configs/trackers/botsort-reid.yaml` |
| conf / IoU / imgsz / classes | `0.25 /0.7 /960 /[2, 5, 7]` |
| device | `0` |

| So sánh | HOTA | DetA | AssA | LocA | IDF1 | MOTA | MOTP | FP | FN | IDSW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bạn vs gold |0.800   |0.777   |0.827   |0.888   |0.937   |0.867   |0.878      |69       |7       |0 |
| ByteTrack control vs gold |0.709   |0.649   |0.776   |0.846   |0.875   |0.749   |0.823      |88      |54       |2 |
| BoT-SORT + ReID vs gold |0.763   |0.711   |0.820   |0.872   |0.900   |0.792   |0.860      |91      |26       |2 |
| ReID vs bạn |0.767   |0.713   |0.826   |0.918   |0.870   |0.745   |0.912      |81      |78       |3 |

## 5. Phân tích — năm câu hỏi

**1. MOTA của bạn cao hơn hay thấp hơn IDF1? Nếu MOTA cao mà IDF1 thấp thì điều đó nói gì, và vì sao MOTA không phạt nặng lỗi ID?**

`MOTA = 0.867; IDF1 = 0.937 => IDF1 cao hơn MOTA.`
`MOTA cao nhưng IDF1 thấp thì nghĩa là detector bắt được phần lớn object (FP/FN ít), nhưng tracker thường xuyên gán sai ID hoặc đổi ID sau occlusion.`
`ID switch chỉ là một thành phần đếm lỗi đơn giản, trong khi IDF1 đánh giá toàn bộ quá trình duy trì identity xuyên suốt sequence. Một ID switch có thể kéo giảm đáng kể IDF1 nhưng chỉ làm MOTA giảm một lượng nhỏ.`

**2. ByteTrack control và BoT-SORT + ReID treatment khác nhau thế nào ở IDF1, AssA và IDSW? Dẫn một frame sequence để giải thích treatment tốt hơn, tệ hơn hoặc không đổi đáng kể. Nhắc rõ đây không cô lập causal effect của ReID vì hai tracker implementation khác.**

IDF1:
ByteTrack: 0.875;
BoT-SORT + ReID: 0.900
=> tăng 0.025

AssA:
ByteTrack: 0.776;
BoT-SORT + ReID: 0.820
=> tăng 0.044

IDSW:
ByteTrack: 2;
BoT-SORT + ReID: 2
=> không thay đổi
ReID giúp giữ định danh (AssA, IDF1) tốt hơn rõ rệt. Tuy nhiên, lưu ý đây là sự so sánh cấp hệ thống (system-level) nên không thể quy toàn bộ chênh lệch này là do ReID, vì implementation của ByteTrack và BoT-SORT là khác nhau. (Sequence ví dụ: Track 3 và 6 bị tách làm 2 mảnh ở ByteTrack nhưng BoT-SORT nối liền được tốt hơn nhờ đặc trưng hình ảnh).


**3. DetA, FP và FN đổi thế nào? Lỗi còn lại là detector hay association?**

DetA
Bạn: 0.777
ByteTrack: 0.649
BoT-SORT: 0.711
Kết quả của 'bạn' có DetA cao nhất.

FP/FN
Bạn:
FP = 69
FN = 7

ByteTrack:
FP = 88
FN = 54

BoT-SORT:
FP = 91
FN = 26

Nhận xét:
FN của 'bạn' cực thấp.
Detector gần như không bỏ sót object.
IDSW = 0.

Do đó: Phần lớn lỗi còn lại không nằm ở association.

Evidence:

AssA = 0.827 rất cao.
IDF1 = 0.937 rất cao.
IDSW = 0.

Lỗi còn lại chủ yếu đến từ:
    Một số FP dư thừa.
    Box localization chưa hoàn hảo (LocA chưa đạt 1.0).

Nói ngắn gọn:
    Bottleneck hiện tại thiên về detection/localization hơn là identity association.

**4. Một chỗ bạn đúng và ReID sai (frame, ID, vì sao):**

`Tại frame 97, ID 7 của ReID đánh nhầm vào sạp hàng bên đường, em đã nhận ra nên không đánh nhãn vào vị trí này.`

**5. Một chỗ ReID làm bạn xem lại annotation (frame, ID, vì sao), hoặc lý do evidence cho thấy model sai:**

`Tại frame 97, ID 9 của ReID, ở đằng sau xe buýt có 2 xe con bị che 1 phần nhưng vẫn có dấu hiệu để em nhìn ra. Em đã phải kiểm tra lại khả năng bản thân đã đánh dấu quá sớm. Tuy nhiên khi đối chiếu với guideline-mini, em thấy guide 'bị che -> chỉ bbox phần nhìn thấy' nên em giữ quyết định vẫn đánh nhãn 2 xe này.`

## 6. Nếu phải gán thêm 10 clip nữa

Bạn sẽ sửa gì trong `GUIDELINE_MINI.md`, và đổi gì trong quy trình làm việc của mình?

`Cần quy định rõ những bộ phận 'trồi ra ngoài' khung xe (vd: gương của xe buýt, xe tải) có tính vào bbox hay không.`

## 7. Tệp đã nộp

- [X] `annotations/clip_01/gt.txt`
- [X] `annotations/clip_02/gt.txt`
- [X] `evidence/pre-gold/clip_01/gt.txt` và `manifest.json`
- [X] `GUIDELINE_MINI.md` đã điền
- [X] `outputs/eval_vs_gold.json`
- [X] `outputs/model_bytetrack_clip_01.txt`
- [x] `outputs/model_reid_clip_01.txt`
- [X] `outputs/model_run_config.json`
- [X] `outputs/eval_bytetrack_vs_gold.json`, `outputs/eval_reid_vs_gold.json`, `outputs/eval_reid_vs_me.json`
- [X] `reports/review_partner.md`
- [X] `reports/REPORT.md` (file này)
