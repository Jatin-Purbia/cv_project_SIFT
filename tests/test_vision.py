"""Synthetic ground-truth checks for the independently implemented algorithms."""
import json
from pathlib import Path
import unittest

import numpy as np

from vision import (Features, Matches, Detection, match_descriptors, fit_affine, transform,
                    ransac_affine, polygon_iou, nms, hough_clusters, detect)

CFG = json.loads((Path(__file__).resolve().parents[1] / 'config.json').read_text())


class GeometryTests(unittest.TestCase):
    def test_matching_agrees_with_brute_force_and_reuses_template(self):
        rng = np.random.default_rng(4)
        t = rng.normal(size=(20, 128))
        s = np.vstack([t[:8]+0.01, t[:8]-0.02, rng.normal(size=(5, 128))])
        d = np.linalg.norm(s[:, None]-t[None], axis=2)
        order = np.argsort(d, axis=1)
        ratios = d[np.arange(len(s)), order[:, 0]] / d[np.arange(len(s)), order[:, 1]]
        expected = np.flatnonzero(ratios < 0.78)
        actual = match_descriptors(s, t, chunk_size=3)
        np.testing.assert_array_equal(actual.scene, expected)
        np.testing.assert_array_equal(actual.template, order[expected, 0])
        np.testing.assert_allclose(actual.ratios, ratios[expected], atol=1e-10)
        self.assertGreater(np.sum(actual.template == 0), 1)

    def test_matching_empty_and_ambiguous(self):
        self.assertEqual(len(match_descriptors(np.zeros((2, 128)), np.zeros((2, 128))).scene), 0)
        self.assertEqual(len(match_descriptors(np.zeros((2, 128)), np.zeros((1, 128))).scene), 0)
        self.assertEqual(len(match_descriptors(np.empty((0, 128)), np.zeros((2, 128))).scene), 0)

    def test_least_squares_and_degeneracy(self):
        source = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], float)
        affine = np.array([[1.1, -0.2, 45], [0.3, 0.8, -3]])
        np.testing.assert_allclose(fit_affine(source, transform(source, affine)), affine, atol=1e-12)
        self.assertIsNone(fit_affine(np.array([[0, 0], [1, 1], [2, 2]]), source[:3]))
        self.assertIsNone(fit_affine(source[:3], np.zeros((3, 2))))

    def test_ransac_recovers_transform_with_majority_outliers(self):
        rng = np.random.default_rng(81)
        source = rng.uniform(0, 250, (100, 2))
        affine = np.array([[0.9, -0.3, 150], [0.3, 0.9, 40]])
        target = transform(source, affine) + rng.normal(0, 0.3, (100, 2))
        target[40:] = rng.uniform(0, 600, (60, 2))
        result = ransac_affine(source, target, CFG, rng)
        self.assertIsNotNone(result)
        recovered, ids, _ = result
        self.assertGreaterEqual(len(ids), 39)
        self.assertLess(np.max(np.linalg.norm(transform(source[:40], recovered)-transform(source[:40], affine), axis=1)), 0.5)

    def test_polygon_iou_and_nms(self):
        a = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
        b = a + [1, 0]
        self.assertAlmostEqual(polygon_iou(a, b), 1/3)
        self.assertAlmostEqual(polygon_iou(a[::-1], b), 1/3)
        self.assertAlmostEqual(polygon_iou(a, a), 1)
        self.assertEqual(polygon_iou(a, a+10), 0)
        self.assertEqual(polygon_iou(a, a+[2, 0]), 0)
        diamond = np.array([[1, 0], [2, 1], [1, 2], [0, 1]])
        self.assertAlmostEqual(polygon_iou(a, diamond), 0.5)
        ds = [Detection(np.eye(2, 3), p, np.arange(n), 1, 1, 1) for p, n in [(a, 10), (a+0.1, 8), (a+10, 9)]]
        kept = nms(ds, 0.45)
        self.assertEqual([len(d.inliers) for d in kept], [10, 9])

    def test_hough_wraparound_and_multiple_instances(self):
        rng = np.random.default_rng(6)
        p = rng.uniform([10, 10], [180, 250], (25, 2))
        tf = Features(p, np.ones(25)*5, np.zeros(25), np.zeros((25, 128)))
        affines = []
        for angle, scale, shift in [(0, 0.9, [50, 50]), (0.5, 0.7, [500, 100]), (-0.7, 1.1, [600, 500])]:
            c, s = np.cos(angle), np.sin(angle)
            affines.append(np.column_stack((scale*np.array([[c, -s], [s, c]]), shift)))
        q = np.vstack([transform(p, a) for a in affines])
        angles = np.repeat([0, 0.5, -0.7], 25) + np.tile(np.linspace(-0.01, 0.01, 25), 3)
        sf = Features(q, np.repeat([4.5, 3.5, 5.5], 25), angles, np.zeros((75, 128)))
        matches = Matches(np.arange(75), np.tile(np.arange(25), 3), np.ones(75)*0.2, np.ones(75))
        groups = hough_clusters(tf, sf, matches, np.arange(75), np.array([99.5, 149.5]), CFG)
        self.assertTrue(any(set(range(25)).issubset(set(g)) for g in groups))
        found, _ = detect(tf, sf, matches, (300, 200), (1000, 1200), CFG)
        self.assertEqual(len(found), 3)
        for affine in affines:
            self.assertTrue(any(np.allclose(d.affine, affine, atol=1e-8) for d in found))

    def test_no_matches_means_no_detections(self):
        f = Features(np.empty((0, 2)), np.array([]), np.array([]), np.empty((0, 128)))
        m = Matches(np.array([], int), np.array([], int), np.array([]), np.array([]))
        found, _ = detect(f, f, m, (100, 100), (500, 500), CFG)
        self.assertEqual(found, [])


if __name__ == '__main__':
    unittest.main()
