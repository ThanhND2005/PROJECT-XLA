import unittest
import numpy as np
from glcm import analyze, DEMO_IMAGE
from glcm_library import analyze_library


class LibraryTests(unittest.TestCase):
    def test_matches_manual_implementation(self):
        rng = np.random.default_rng(42)
        for image in [DEMO_IMAGE, [[0] * 8 for _ in range(7)],
                      rng.integers(0, 4, size=(9, 11)).tolist()]:
            for symmetric in (False, True):
                expected = analyze(image, 4, (1, 2, 3), symmetric=symmetric)
                actual = analyze_library(image, 4, (1, 2, 3), symmetric=symmetric)
                for a, b in zip(expected['results'], actual['results']):
                    self.assertEqual(a['counts'], b['counts'])
                    for key in a['features']:
                        self.assertAlmostEqual(a['features'][key], b['features'][key], places=12)

    def test_no_valid_pairs(self):
        with self.assertRaises(ValueError):
            analyze_library([[0]], levels=2)


if __name__ == '__main__':
    unittest.main()
