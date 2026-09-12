"""Run: python run_pipeline.py --config config.json"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import cv2
import numpy as np

from vision import Features, match_descriptors, detect


def extract(image, contrast):
    sift = cv2.SIFT_create(contrastThreshold=contrast)
    keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
    return Features(np.array([k.pt for k in keypoints], float).reshape(-1, 2),
                    np.array([k.size for k in keypoints]),
                    np.deg2rad([k.angle for k in keypoints]),
                    descriptors if descriptors is not None else np.empty((0, 128), np.float32))


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Could not write {path}')


def matching_overlay(template, scene, tf, sf, matches, limit=160):
    """Deterministically subsample tentative matches for a readable comparison."""
    h = max(template.shape[0], scene.shape[0])
    w = template.shape[1]
    canvas = np.full((h, w + scene.shape[1], 3), 245, np.uint8)
    canvas[:template.shape[0], :w] = template
    canvas[:scene.shape[0], w:] = scene
    rng = np.random.default_rng(1)
    ids = np.sort(rng.choice(len(matches.scene), min(limit, len(matches.scene)), replace=False))
    for i in ids:
        start = tuple(np.rint(tf.points[matches.template[i]]).astype(int))
        end = tuple(np.rint(sf.points[matches.scene[i]] + [w, 0]).astype(int))
        color = tuple(int(v) for v in rng.integers(50, 240, 3))
        cv2.line(canvas, start, end, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, end, 3, color, -1, cv2.LINE_AA)
    return canvas, len(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--template', help='Override template path (relative to working directory)')
    parser.add_argument('--scene', help='Override scene path (relative to working directory)')
    parser.add_argument('--output', help='Override output directory')
    parser.add_argument('--roi', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'))
    parser.add_argument('--full-template', action='store_true', help='Disable configured template crop')
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    base = config_path.parent
    template_path = Path(args.template).resolve() if args.template else base / cfg['template']
    scene_path = Path(args.scene).resolve() if args.scene else base / cfg['scene']
    output = Path(args.output).resolve() if args.output else base / cfg['output']
    original, scene = cv2.imread(str(template_path)), cv2.imread(str(scene_path))
    if original is None or scene is None:
        raise SystemExit(f'Could not read input photographs: {template_path}, {scene_path}')
    roi = None if args.full_template else (args.roi or cfg.get('template_roi'))
    if roi is not None:
        x, y, w, h = roi
        if min(x, y) < 0 or min(w, h) <= 0 or x+w > original.shape[1] or y+h > original.shape[0]:
            raise SystemExit('ROI must be a positive rectangle within the template photograph')
        template = original[y:y+h, x:x+w].copy()
    else:
        template = original.copy()
    cfg['template_roi'] = roi
    start = time.perf_counter()
    tf, sf = extract(template, cfg['sift_contrast']), extract(scene, cfg['sift_contrast'])
    extraction_time = time.perf_counter() - start
    matches = match_descriptors(sf.descriptors, tf.descriptors, cfg['ratio'])
    matching_time = time.perf_counter() - start - extraction_time
    detections, diagnostics = detect(tf, sf, matches, template.shape, scene.shape, cfg)
    total_time = time.perf_counter() - start
    output.mkdir(parents=True, exist_ok=True)
    write_image(output / 'template_crop.jpg', template)
    overlay, shown = matching_overlay(template, scene, tf, sf, matches)
    write_image(output / 'naive_matches.jpg', overlay)
    final = scene.copy()
    colors = [(70, 220, 30), (0, 190, 255), (255, 170, 50), (220, 70, 230)]
    records = []
    for i, detection in enumerate(detections, 1):
        polygon = np.rint(detection.corners).astype(np.int32)
        color = colors[(i-1) % len(colors)]
        cv2.polylines(final, [polygon], True, color, 3, cv2.LINE_AA)
        anchor = polygon[np.argmin(polygon[:, 1])]
        label = (int(np.clip(anchor[0], 5, scene.shape[1]-160)), int(max(25, anchor[1]-12)))
        cv2.putText(final, f'Object {i}', label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(final, f'Object {i}', label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        record = asdict(detection)
        record = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in record.items()}
        record['id'], record['inlier_count'] = i, len(detection.inliers)
        records.append(record)
    write_image(output / 'detections.jpg', final)
    results = {
        'template': str(template_path), 'scene': str(scene_path),
        'coordinate_system': 'Affine maps cropped-template pixels to original scene pixels; corners are un-clipped.',
        'config': cfg, 'template_keypoints': len(tf.points), 'scene_keypoints': len(sf.points),
        'tentative_matches': len(matches.scene), 'matches_shown': shown,
        'detection_count': len(detections), 'detections': records, 'diagnostics': diagnostics,
        'timing_seconds': {'sift': extraction_time, 'matching': matching_time,
                           'geometry': total_time-extraction_time-matching_time, 'total': total_time},
        'versions': {'opencv': cv2.__version__, 'numpy': np.__version__},
    }
    (output / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'SIFT: {len(tf.points)} template / {len(sf.points)} scene; {len(matches.scene)} ratio-test matches')
    for record in records:
        print(f"Object {record['id']}: {record['inlier_count']} inliers, median error {record['median_error']:.2f}px, coverage {record['coverage']:.1%}")
    print(f'{len(detections)} detections in {total_time:.2f}s. Outputs: {output}')


if __name__ == '__main__':
    main()
