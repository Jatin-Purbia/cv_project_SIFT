"""Local web UI for the detection pipeline. Run: python app.py

Reuses extract()/matching_overlay() from run_pipeline.py and match_descriptors()/
detect() from vision.py directly, so the UI runs the exact same algorithm as the CLI.
"""
import base64
import json
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from run_pipeline import extract, matching_overlay
from vision import match_descriptors, detect

BASE = Path(__file__).resolve().parent
DEFAULT_CFG = json.loads((BASE / 'config.json').read_text(encoding='utf-8'))

# BGR (OpenCV) draw colors and their RGB hex equivalents, kept in sync so the
# frontend can color-code each detection card to match its box in the image.
COLORS_BGR = [(70, 220, 30), (0, 190, 255), (255, 170, 50), (220, 70, 230)]
COLORS_HEX = ['#1edc46', '#ffbe00', '#32aaff', '#e646dc']

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 25 MB total request body


def decode_image(file_storage):
    data = np.frombuffer(file_storage.read(), np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def encode_image(image):
    ok, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise OSError('Could not encode result image')
    return base64.b64encode(buf).decode('ascii')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/detect', methods=['POST'])
def api_detect():
    template_file = request.files.get('template')
    scene_file = request.files.get('scene')
    if not template_file or not scene_file:
        return jsonify(error='Please provide both an object image and a scene image.'), 400

    template = decode_image(template_file)
    scene = decode_image(scene_file)
    if template is None or scene is None:
        return jsonify(error='One of the uploaded files could not be read as an image.'), 400

    cfg = dict(DEFAULT_CFG)
    cfg['template_roi'] = None  # uploaded object photo is used as-is, no cropping

    try:
        tf = extract(template, cfg['sift_contrast'])
        sf = extract(scene, cfg['sift_contrast'])
        matches = match_descriptors(sf.descriptors, tf.descriptors, cfg['ratio'])
        detections, diagnostics = detect(tf, sf, matches, template.shape, scene.shape, cfg)
    except Exception as exc:  # surface pipeline errors to the UI instead of a raw 500
        return jsonify(error=f'Detection failed: {exc}'), 500

    final = scene.copy()
    records = []
    for i, d in enumerate(detections, 1):
        color = COLORS_BGR[(i - 1) % len(COLORS_BGR)]
        polygon = np.rint(d.corners).astype(np.int32)
        cv2.polylines(final, [polygon], True, color, 3, cv2.LINE_AA)
        anchor = polygon[np.argmin(polygon[:, 1])]
        label_pos = (int(np.clip(anchor[0], 5, scene.shape[1] - 160)), int(max(25, anchor[1] - 12)))
        cv2.putText(final, f'Object {i}', label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(final, f'Object {i}', label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        records.append({
            'id': i,
            'inliers': len(d.inliers),
            'median_error': round(float(d.median_error), 2),
            'coverage': round(float(d.coverage) * 100, 1),
            'color': COLORS_HEX[(i - 1) % len(COLORS_HEX)],
        })

    overlay, shown = matching_overlay(template, scene, tf, sf, matches)

    return jsonify(
        detections=records,
        detection_image=encode_image(final),
        match_image=encode_image(overlay),
        stats={
            'template_keypoints': int(len(tf.points)),
            'scene_keypoints': int(len(sf.points)),
            'tentative_matches': int(len(matches.scene)),
            'matches_shown': int(shown),
            'detection_count': len(detections),
        },
    )


if __name__ == '__main__':
    app.run(debug=True)
