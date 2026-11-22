# Cài đặt GLCM bằng Python

## Texture Lab

### Docker

Mở Docker Desktop ở chế độ Linux containers rồi chạy từ thư mục dự án.

```powershell
docker compose up --build -d
```

Mở http://127.0.0.1:8000. Khi build, Docker tải ImageJ 1.54p từ kho cập nhật Fiji và plugin GLCM Texture 0.4. Container có Java, màn hình ảo Xvfb, toàn bộ 800 ảnh, hai CSV đặc trưng và web. Hai model được huấn luyện từ CSV khi container khởi động.

Build kiểm tra cả hai nhánh trích xuất và dự đoán trên cùng một ảnh từ dataset. Muốn xem trạng thái và log sau khi chạy, dùng `docker compose ps` và `docker compose logs texture-lab`. Nếu cổng 8000 đang được sử dụng, đặt `$env:GLCM_PORT = '8080'` trước khi chạy Compose và mở http://127.0.0.1:8080.

### Chạy trực tiếp bằng Python

Chạy từ thư mục dự án trong PowerShell.

```powershell
python -m pip install -r requirements-web.txt
python web_app.py
```

Mở http://127.0.0.1:8000. Có thể tải ảnh lên, kéo thả hoặc chọn một trong năm ảnh mẫu.

- **Dự đoán ảnh** hiển thị lớp Chăn, Sàn, Cỏ, Gạo hoặc Cát và xác suất ước tính của từng lớp dưới dạng phần trăm. Máy chủ huấn luyện hai model riêng khi khởi động từ hai CSV Python và Fiji. Mỗi model dùng 600 ảnh thuộc nhóm a/b/c. Các ảnh mẫu trên web thuộc nhóm d.
- **Trích xuất đặc trưng** hiển thị Contrast, Correlation, ASM, Energy, Homogeneity và Entropy. Với Python, có thể mở bản đồ nhiệt và tải CSV ma trận xác suất GLCM 16 × 16. Macro Fiji xuất sáu giá trị từ plugin.
- Cả hai phương pháp chạy ở 16 mức xám, khoảng cách 1, góc 0°. Python gọi trực tiếp thuật toán tự viết trong `glcm.py`. Model dùng năm đặc trưng, với Energy được suy ra từ ASM.

### Kết nối Fiji

Cài [Fiji](https://imagej.net/software/fiji/downloads), chép `GLCM_Texture.class` từ [GLCM Texture 0.4](https://wsr.imagej.net/ij/plugins/texture.html) vào thư mục `plugins` và kiểm tra mục **Plugins → GLCM Texture** trong Fiji. Đặt đường dẫn tới tệp thực thi Fiji trước khi khởi động máy chủ.

```powershell
$env:GLCM_FIJI_EXE = 'C:\Fiji.app\ImageJ-win64.exe'
python web_app.py
```

Thay ví dụ trên bằng đường dẫn thực tế, ví dụ `fiji.exe` tùy bản cài. Web chuyển ảnh sang xám 8 bit rồi chạy macro `GLCM_Texture_Batch_L16_d1_a0.ijm` qua Fiji. Mặc định chạy headless. Nếu bản Fiji hoặc plugin của bạn cần ngữ cảnh đồ họa, thử đặt `$env:GLCM_FIJI_MODE = 'batch'` trước khi khởi động lại máy chủ trên máy có desktop.

## Tái lập thực nghiệm Chương 2

Chạy `python -X utf8 chapter2_experiment.py` từ thư mục dự án. Chương trình dùng
duy nhất `dataset_xla/rice1-without-rotations/rice1-a-p001.png`, chuyển xám
và cắt vùng giữa 128×128 pixel. Kết quả gồm ba bảng khảo sát G = 8, 16, 32, 64;
d = 1, 2, 4; bốn hướng, cùng bảng đối chiếu scikit-image trong
`chapter2_results.json`. Hai ảnh minh họa quá trình tiền xử lý được lưu ở
`chapter2_rice_gray.png` và `chapter2_rice_quantized_g16.png`.

## Bản dùng thư viện scikit-image

`glcm_library.py` dùng `graycomatrix` để đếm cặp và `graycoprops` để tính cả
sáu đặc trưng (bao gồm entropy, yêu cầu scikit-image >= 0.25).
Chỉ dùng chung phần kiểm tra đầu vào, lượng tử hóa, CLI và vẽ báo cáo với bản tự viết.

```powershell
python -m pip install -r requirements-library.txt
python -X utf8 glcm_library.py --angles 0
python -X utf8 glcm_library.py dataset_xla\floor1-without-rotations\floor1-a-p001.png --levels 16 --distances 1 2 --output floor_library.json --output-image floor_library.png
python -X utf8 -m unittest test_glcm test_glcm_library -v
```

Nếu không chỉ định `--output-image`, ảnh báo cáo có hậu tố `_glcm_library.png`.
Để giữ đúng quy ước tài liệu, bản thư viện dùng góc âm và khoảng cách
Euclid `sqrt(2)*d` cho góc chéo, thay vì truyền nguyên d cho mọi hướng.
Bộ kiểm thử so sánh chính xác ma trận đếm và sáu đặc trưng với bản tự viết
ở bốn hướng, d=1,2,3, cả hai chế độ đối xứng; gồm ảnh ví dụ, ảnh đồng nhất
và ảnh ngẫu nhiên cố định seed.

Tham khảo API: https://scikit-image.org/docs/stable/api/skimage.feature.html

Triển khai theo mục 2 và 3 của `BTL-XLA-N15.md`, không dùng hàm GLCM có sẵn. Python thuần xử lý ma trận; Pillow chỉ dùng để đọc ảnh và chuyển sang ảnh xám 8-bit.

## Chạy

```powershell
python glcm.py --angles 0
python -m pip install -r requirements.txt
python glcm.py anh.png --levels 16 --distances 1 2 --angles 0 45 90 135 --output ketqua.json
python -m unittest -v
```

Không truyền ảnh sẽ chạy ma trận 4×4 ở mục 3.1.5, dùng G=4 (không lượng tử hóa lại). Mặc định xét bốn hướng, d=1, đối xứng hóa. `--no-symmetric` tắt đối xứng hóa. JSON chứa ma trận đếm, xác suất, đặc trưng từng hướng và trung bình đặc trưng theo hướng cho từng khoảng cách.

## Dùng trong chương trình

Khi truyền file ảnh, chương trình tự lưu báo cáo PNG cạnh ảnh đầu vào, tên
`<tên ảnh>_glcm.png`. Báo cáo gồm ảnh xám, ảnh lượng tử hóa (giãn mức sáng
để hiển thị), heatmap xác suất từng hướng/khoảng cách và sáu đặc trưng.
Các heatmap dùng chung thang màu tuyến tính để so sánh. Đây là ảnh báo cáo
thống kê GLCM toàn ảnh, không phải ảnh phân đoạn hay bản đồ texture cục bộ.
Nhãn trong ảnh dùng tiếng Anh để không phụ thuộc font tiếng Việt.
File báo cáo cùng tên được ghi lại khi chạy lần sau.

```powershell
python glcm.py glcm_sample.png --distances 1 2 --output-image baocao.png
python glcm.py --angles 0 --output-image demo_glcm.png
```

```python
from glcm import quantize, compute_glcm, texture_features, analyze

gray = [[0, 64, 128], [64, 128, 255]]
q = quantize(gray, levels=4)
p = compute_glcm(q, levels=4, distance=1, angle=0)
print(texture_features(p))
report = analyze(q, levels=4)
```

API nhận ma trận list các số nguyên. `quantize` nhận mức xám 0–255; `compute_glcm` và `analyze` nhận mức xám đã lượng tử hóa 0–G−1. Không đưa trực tiếp ảnh 16-bit hoặc giá trị thực vào API này.

## Quy ước và công thức

- Lượng tử hóa: `floor(I * G / 256)`, với 2 ≤ G ≤ 256.
- Offset `(hàng, cột)`: 0° `(0,d)`, 45° `(-d,d)`, 90° `(-d,0)`, 135° `(-d,-d)`. Hướng 45° là lên phải theo vector trong tài liệu; phần mô tả “chéo bên trái” ở hàng đó bị nhầm. d là bước trên lưới; đường chéo có khoảng cách Euclid √2·d.
- Bỏ qua cặp vượt biên. Đối xứng bằng `C + Cᵀ`, sau đó chia tổng để có p. Nếu không có cặp hợp lệ thì báo lỗi.
- Contrast: Σ `(i-j)² p(i,j)`.
- Homogeneity: Σ `p(i,j)/(1+(i-j)²)`.
- ASM: Σ `p(i,j)²`; Energy: `sqrt(ASM)`. Hai đại lượng được trả riêng để tránh cách gọi lẫn trong tài liệu.
- Correlation: hiệp phương sai chia tích độ lệch chuẩn của hai phân phối biên; quy ước bằng 1 khi một phương sai bằng 0.
- Entropy: `-Σ p ln(p)`, bỏ các phần tử 0, đơn vị nat.
- Lấy trung bình **đặc trưng** của các hướng, không tính đặc trưng từ ma trận trung bình. Cách này giảm phụ thuộc hướng nhưng không bảo đảm bất biến với mọi góc quay.

Ví dụ tài liệu ở d=1, góc 0°: Contrast ≈ 0.583333, Correlation ≈ 0.719533, ASM ≈ 0.145833, Energy ≈ 0.381881, Homogeneity ≈ 0.808333, Entropy ≈ 2.094729. Các bài kiểm tra đối chiếu ma trận và giá trị này, ảnh sọc, hướng chéo, khoảng cách, lượng tử hóa và trường hợp biên.

Python thuần thuận tiện để học và kiểm tra từng bước; ảnh lớn sẽ xử lý chậm hơn phiên bản vector hóa.
