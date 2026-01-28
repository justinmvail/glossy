#!/usr/bin/env python3
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
