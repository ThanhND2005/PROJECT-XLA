"""Exercise both Docker analysis pipelines against one original dataset image."""

import csv
import math

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


if __name__ == "__main__":
    main()
