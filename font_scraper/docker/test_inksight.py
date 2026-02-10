#!/usr/bin/env python3
"""Test script for InkSight vectorization inside Docker container.

This script is designed to run inside the InkSight Docker container to verify
that the model loading and inference pipeline works correctly. It tests the
full vectorization workflow from rendering a font to extracting strokes.

Docker Usage:
    This script is typically executed via docker run::

        docker run --rm --gpus all \\
            -v /home/server/inksight/model:/app/model:ro \\
            -v /path/to/font_scraper:/app/workspace:ro \\
            inksight:latest \\
            python3 /app/workspace/docker/test_inksight.py

Volume Mounts Required:
    - /app/model: InkSight saved_model directory (read-only)
    - /app/workspace: font_scraper directory containing inksight_vectorizer.py

Output:
    - Prints stroke count and point statistics to stdout
    - Saves visualization to /tmp/inksight_docker_hello.png inside container

Example Output:
    Strokes: 5, Points: 127
    Saved to /tmp/inksight_docker_hello.png

Note:
    This script suppresses TensorFlow logging by setting TF_CPP_MIN_LOG_LEVEL=3
    to reduce noise during testing. Remove this for debugging TensorFlow issues.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.insert(0, '/app/workspace')

from inksight_vectorizer import InkSightVectorizer, visualize

v = InkSightVectorizer(model_path='/app/model')
v.load_model()

result = v.process('/app/workspace/fonts/dafont/Braydon Script.ttf', 'hello')
visualize(result, output_path='/tmp/inksight_docker_hello.png')
print(f'Strokes: {len(result.strokes)}, Points: {sum(len(s.points) for s in result.strokes)}')
print('Saved to /tmp/inksight_docker_hello.png')
