import math
import unittest

from glcm import DEMO_IMAGE, compute_glcm, quantize, texture_features


class GLCMTests(unittest.TestCase):
    def test_document_example(self):
        counts = compute_glcm(DEMO_IMAGE, 4, symmetric=False, normalized=False)
        self.assertEqual(counts, [[2, 2, 1, 0], [0, 2, 0, 0], [0, 0, 3, 1], [0, 0, 0, 1]])
        p = compute_glcm(DEMO_IMAGE, 4)
        self.assertAlmostEqual(sum(map(sum, p)), 1)
        f = texture_features(p)
        for key, value in dict(contrast=14/24, correlation=431/599,
                               asm=84/576, energy=math.sqrt(84/576),
                               homogeneity=19.4/24).items():
            self.assertAlmostEqual(f[key], value)
        self.assertAlmostEqual(f['entropy'], 2.0947, places=4)

    def test_stripes(self):
        rows = [[0, 1, 0, 1]] * 4
        horizontal = texture_features(compute_glcm(rows, 2, angle=0))
        vertical = texture_features(compute_glcm(rows, 2, angle=90))
        self.assertEqual(horizontal['contrast'], 1)
        self.assertEqual(horizontal['correlation'], -1)
        self.assertEqual(vertical['contrast'], 0)
        self.assertEqual(vertical['correlation'], 1)
        self.assertAlmostEqual(horizontal['entropy'], math.log(2))

    def test_offsets_against_manual_pairs(self):
        rows = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        for angle, pairs in {45: [(3, 1), (4, 2), (6, 4), (7, 5)],
                             135: [(4, 0), (5, 1), (7, 3), (8, 4)]}.items():
            c = compute_glcm(rows, 9, angle=angle, symmetric=False, normalized=False)
            self.assertEqual(sum(map(sum, c)), 4)
            for i, j in pairs:
                self.assertEqual(c[i][j], 1)
        c = compute_glcm(rows, 9, distance=2, symmetric=False, normalized=False)
        self.assertEqual(sum(map(sum, c)), 3)
        self.assertEqual(c[0][2], 1)

    def test_constant_quantization_and_invalid_input(self):
        self.assertEqual(quantize([[0, 63, 64, 255]], 4), [[0, 0, 1, 3]])
        f = texture_features(compute_glcm([[0, 0]], 2))
        self.assertEqual((f['energy'], f['correlation'], f['entropy']), (1, 1, 0))
        for image in ([], [[0], [0, 1]], [[-1]], [[256]]):
            with self.assertRaises(ValueError):
                quantize(image)
        with self.assertRaises(ValueError):
            compute_glcm([[0]], 2)
        with self.assertRaises(ValueError):
            texture_features([[0, 0], [0, 0]])


if __name__ == '__main__':
    unittest.main()
