"""Exercise both Docker analysis pipelines against one original dataset image."""

import csv
import io
import math

from PIL import Image

from web_app import FEATURE_NAMES, ROOT, SOURCES, find_fiji, load_models, run_analysis, sample_path


def main():
    sample = sample_path("grass")
    data = sample.read_bytes()
    image_name = sample.relative_to(ROOT / "dataset_xla").as_posix()
    models = load_models()
    fiji = find_fiji()
    for source, path in SOURCES.items():
        method = source.lower()
        with path.open(encoding="utf-8-sig", newline="") as stream:
            expected = next(row for row in csv.DictReader(stream) if row["image"] == image_name)
        extraction = run_analysis(data, method, False, models, fiji)
        for name in FEATURE_NAMES:
            actual, reference = extraction["features"][name], float(expected[name])
            if not math.isclose(actual, reference, rel_tol=0, abs_tol=0.000002):
                raise ValueError(f"{source} {name}: {actual} != {reference}")
        prediction = run_analysis(data, method, True, models, fiji)
        if prediction["ranking"][0]["id"] != "grass":
            raise ValueError(f"{source} returned an unexpected class")
        print(f"{source}: extraction and prediction passed.")

    gray = Image.new("L", (1, 4))
    gray.putdata([255, 36, 37, 109])
    with io.BytesIO() as stream:
        gray.save(stream, format="PNG")
        data = stream.getvalue()
    # Fiji includes pairs with its zero-valued border pixels.
    for method, contrast in (("python", 14.5), ("fiji", 16.25)):
        result = run_analysis(data, method, False, models, fiji, levels=7, distance=2, angle=90)
        if not math.isclose(result["features"]["contrast"], contrast, abs_tol=0.000002):
            raise ValueError(f"{method} returned unexpected contrast with G=7, d=2, angle=90")
        print(f"{method}: configurable extraction passed.")


if __name__ == "__main__":
    main()
