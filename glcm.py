import argparse
import json
import math
from pathlib import Path


def _integer(value, name, low, high=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < low or (high is not None and value > high):
        raise ValueError(f'{name} phải là số nguyên trong [{low}, {high or "∞"}].')


def _image(image, levels):
    rows = [list(row) for row in image]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('Ảnh phải là ma trận chữ nhật không rỗng.')
    for row in rows:
        for value in row:
            _integer(value, 'Mức xám', 0, levels - 1)
    return rows


def quantize(image, levels=16):
    _integer(levels, 'levels', 2, 256)
    return [[value * levels // 256 for value in row] for row in _image(image, 256)]


def compute_glcm(image, levels=16, distance=1, angle=0, symmetric=True, normalized=True):
    """Nhận ảnh ĐÃ lượng tử hóa. Trả ma trận G x G dạng list.

    Offset (hàng, cột): 0°=(0,d), 45°=(-d,d), 90°=(-d,0),
    135°=(-d,-d). Bỏ qua cặp vượt biên, không đệm ảnh.
    """
    _integer(levels, 'levels', 2, 256)
    _integer(distance, 'distance', 1)
    if angle not in (0, 45, 90, 135):
        raise ValueError('angle phải là 0, 45, 90 hoặc 135 độ.')
    rows = _image(image, levels)
    height, width = len(rows), len(rows[0])
    dr, dc = {0: (0, distance), 45: (-distance, distance),
              90: (-distance, 0), 135: (-distance, -distance)}[angle]
    counts = [[0] * levels for _ in range(levels)]
    for r in range(height):
        for c in range(width):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                i, j = rows[r][c], rows[nr][nc]
                counts[i][j] += 1
                if symmetric:
                    counts[j][i] += 1  # Đường chéo cũng nhân đôi theo C + Cᵀ.
    total = sum(map(sum, counts))
    if total == 0:
        raise ValueError(f'Không có cặp pixel hợp lệ với d={distance}, angle={angle}.')
    return [[v / total for v in row] for row in counts] if normalized else counts


def texture_features(matrix):
    """Tự chuẩn hóa ma trận đếm/xác suất; entropy đơn vị nat.

    Quy ước correlation=1 khi một trong hai phương sai bằng 0.
    """
    p = [list(row) for row in matrix]
    n = len(p)
    if not n or any(len(row) != n for row in p):
        raise ValueError('GLCM phải là ma trận vuông không rỗng.')
    if any(not math.isfinite(v) or v < 0 for row in p for v in row):
        raise ValueError('GLCM phải chứa số hữu hạn không âm.')
    total = sum(map(sum, p))
    if not math.isfinite(total) or total <= 0:
        raise ValueError('Tổng GLCM phải hữu hạn và dương.')
    p = [[v / total for v in row] for row in p]
    px = list(map(sum, p))
    py = [sum(p[i][j] for i in range(n)) for j in range(n)]
    mx = sum(i * px[i] for i in range(n))
    my = sum(j * py[j] for j in range(n))
    vx = sum((i - mx) ** 2 * px[i] for i in range(n))
    vy = sum((j - my) ** 2 * py[j] for j in range(n))
    contrast = homogeneity = asm = entropy = covariance = 0.0
    for i in range(n):
        for j in range(n):
            value = p[i][j]
            contrast += (i - j) ** 2 * value
            homogeneity += value / (1 + (i - j) ** 2)
            asm += value ** 2
            covariance += (i - mx) * (j - my) * value
            if value > 0:
                entropy -= value * math.log(value)
    correlation = covariance / math.sqrt(vx * vy) if vx > 0 and vy > 0 else 1.0
    return dict(contrast=contrast, correlation=correlation, asm=asm,
                energy=math.sqrt(asm), homogeneity=homogeneity, entropy=entropy)


def analyze(image, levels=16, distances=(1,), angles=(0, 45, 90, 135), symmetric=True):
    distances, angles = tuple(distances), tuple(angles)
    if not distances or not angles:
        raise ValueError('Cần ít nhất một khoảng cách và một hướng.')
    results = []
    means = {}
    for distance in distances:
        group = []
        for angle in angles:
            counts = compute_glcm(image, levels, distance, angle, symmetric, False)
            total = sum(map(sum, counts))
            p = [[v / total for v in row] for row in counts]
            features = texture_features(p)
            group.append(features)
            results.append(dict(distance=distance, angle=angle, counts=counts,
                                probabilities=p, features=features))
        means[str(distance)] = {key: sum(f[key] for f in group) / len(group) for key in group[0]}
    return dict(levels=levels, symmetric=symmetric, results=results, mean_features=means)


DEMO_IMAGE = [[0, 0, 1, 1], [0, 0, 1, 1], [0, 2, 2, 2], [2, 2, 3, 3]]


def save_report_image(gray, quantized, report, destination):
    """Xuất báo cáo PNG; màu heatmap dùng chung thang xác suất tuyến tính."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=15)
    items = report['results']
    canvas = Image.new('RGB', (1120, 470 + 360 * math.ceil(len(items) / 2)), '#f2f5fa')
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 15), 'GLCM TEXTURE ANALYSIS', font=font, fill='#15243b')
    draw.text((24, 44), f"Levels: {report['levels']} | Symmetric: {report['symmetric']} | Entropy: nat", font=small, fill='black')
    q = Image.new('L', (len(quantized[0]), len(quantized)))
    q.putdata([round(v * 255 / (report['levels'] - 1)) for row in quantized for v in row])
    for x, picture, title in ((24, gray, 'Grayscale input'), (570, q, 'Quantized (display scaled to 0-255)')):
        draw.text((x, 78), title, font=font, fill='black')
        preview = ImageOps.contain(picture, (510, 320), method=Image.Resampling.NEAREST)
        canvas.paste(preview, (x, 106))
    peak = max(v for item in items for row in item['probabilities'] for v in row)
    draw.text((24, 441), f'Heatmaps: shared linear scale 0 (black) to {peak:.5f} (yellow). Row=i, column=j.', font=small, fill='black')
    for index, item in enumerate(items):
        x, y = 24 + (index % 2) * 546, 480 + (index // 2) * 360
        draw.text((x, y), f"d={item['distance']} | angle={item['angle']} deg", font=font, fill='black')
        n = report['levels']
        heat = Image.new('RGB', (n, n))
        heat.putdata([(round(255 * v / peak), round(220 * v / peak), 0)
                      for row in item['probabilities'] for v in row])
        canvas.paste(heat.resize((256, 256), Image.Resampling.NEAREST), (x + 20, y + 45))
        draw.text((x + 20, y + 24), f'j: 0 -> {n - 1}', font=small, fill='black')
        draw.text((x + 20, y + 307), f'i: 0 (top) -> {n - 1} (bottom)', font=small, fill='black')
        for k, (name, value) in enumerate(item['features'].items()):
            draw.text((x + 290, y + 50 + k * 34), f'{name}: {value:.5f}', font=small, fill='black')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format='PNG')


def main(analyze_fn=analyze, image_suffix='_glcm'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', nargs='?', help='File ảnh; bỏ trống để chạy ví dụ mục 3.1.5.')
    parser.add_argument('--levels', type=int, default=16)
    parser.add_argument('--distances', type=int, nargs='+', default=[1])
    parser.add_argument('--angles', type=int, nargs='+', default=[0, 45, 90, 135])
    parser.add_argument('--no-symmetric', action='store_true')
    parser.add_argument('--output', type=Path, help='Lưu ma trận và đặc trưng vào JSON.')
    parser.add_argument('--output-image', type=Path, help='Đường dẫn báo cáo PNG (mặc định: <tên ảnh>_glcm.png).')
    args = parser.parse_args()
    try:
        if args.image:
            from PIL import Image
            with Image.open(args.image) as source:
                gray = source.convert('L')
                data = list(gray.tobytes())
                rows = [data[r * gray.width:(r + 1) * gray.width] for r in range(gray.height)]
            levels = args.levels
            rows = quantize(rows, levels)
        else:
            rows, levels = DEMO_IMAGE, 4
        report = analyze_fn(rows, levels, args.distances, args.angles, not args.no_symmetric)
        for item in report['results']:
            print(f"d={item['distance']}, angle={item['angle']}")
            for key, value in item['features'].items():
                print(f'  {key:12s}: {value:.6f}')
        print('Mean features:', json.dumps(report['mean_features'], indent=2))
        if args.output:
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        if args.image or args.output_image:
            if not args.image:
                from PIL import Image
                gray = Image.new('L', (4, 4))
                gray.putdata([v * 85 for row in rows for v in row])
            destination = args.output_image or Path(args.image).with_name(Path(args.image).stem + image_suffix + '.png')
            save_report_image(gray, rows, report, destination)
            print(f'Saved report image: {destination.resolve()}')
    except ImportError:
        parser.error('Đọc ảnh cần Pillow: python -m pip install -r requirements.txt')
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
