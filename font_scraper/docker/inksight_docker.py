#!/usr/bin/env python3
"""
InkSight Docker Wrapper

Run InkSight vectorization in a Docker container with GPU support.
This avoids dependency conflicts between TensorFlow and PyTorch.

Usage:
    from inksight_docker import InkSightDocker

    inksight = InkSightDocker()
    result = inksight.process('fonts/dafont/MyFont.ttf', 'hello')
    print(f"Strokes: {len(result['strokes'])}")
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import base64


class InkSightDocker:
    """Run InkSight in Docker container with GPU."""

    IMAGE = "inksight:latest"
    MODEL_PATH = "/home/server/inksight/model"

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or self.MODEL_PATH
        self._check_docker()

    def _check_docker(self):
        """Verify Docker and GPU are available."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True, text=True, check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Docker not installed or not running")

    def process(
        self,
        font_path: str,
        word: str,
        output_image: Optional[str] = None
    ) -> Dict:
        """
        Vectorize a word using InkSight in Docker.

        Args:
            font_path: Path to font file (.ttf, .otf, .woff2)
            word: Word to render and vectorize
            output_image: Optional path to save visualization

        Returns:
            Dict with 'strokes' (list of point arrays), 'image' (base64 if requested)
        """
        font_path = str(Path(font_path).resolve())
        font_dir = str(Path(font_path).parent)
        font_name = Path(font_path).name

        # Python code to run inside container
        code = f'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import json
import base64
from io import BytesIO

# Add workspace to path
sys.path.insert(0, '/app/workspace')

from inksight_vectorizer import InkSightVectorizer

v = InkSightVectorizer(model_path='/app/model')
v.load_model()

result = v.process('/fonts/{font_name}', '{word}')

# Convert to JSON-serializable format
output = {{
    'strokes': [s.points.tolist() for s in result.strokes],
    'num_strokes': len(result.strokes),
    'total_points': sum(len(s.points) for s in result.strokes),
    'word': result.word
}}

# Save visualization if requested
output_path = '{output_image or ""}'
if output_path:
    from inksight_vectorizer import visualize
    visualize(result, output_path=output_path)
    output['visualization'] = output_path

print(json.dumps(output))
'''

        # Build docker command
        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{self.model_path}:/app/model:ro",
            "-v", f"{font_dir}:/fonts:ro",
            "-v", f"{Path.cwd()}:/app/workspace:ro",
        ]

        # Mount output directory if saving visualization
        if output_image:
            output_dir = str(Path(output_image).parent.resolve())
            cmd.extend(["-v", f"{output_dir}:{output_dir}"])

        cmd.extend([self.IMAGE, "python3", "-c", code])

        # Run container
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"InkSight Docker failed: {result.stderr}")

        # Parse JSON output (last line)
        output_lines = result.stdout.strip().split('\n')
        json_line = output_lines[-1]

        return json.loads(json_line)

    def batch_process(
        self,
        font_path: str,
        words: List[str]
    ) -> List[Dict]:
        """Process multiple words in a single container run (more efficient)."""
        font_path = str(Path(font_path).resolve())
        font_dir = str(Path(font_path).parent)
        font_name = Path(font_path).name
        words_json = json.dumps(words)

        code = f'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import json

sys.path.insert(0, '/app/workspace')
from inksight_vectorizer import InkSightVectorizer

v = InkSightVectorizer(model_path='/app/model')
v.load_model()

words = {words_json}
results = []

for word in words:
    try:
        result = v.process('/fonts/{font_name}', word)
        results.append({{
            'word': word,
            'strokes': [s.points.tolist() for s in result.strokes],
            'num_strokes': len(result.strokes),
            'total_points': sum(len(s.points) for s in result.strokes),
            'success': True
        }})
    except Exception as e:
        results.append({{'word': word, 'success': False, 'error': str(e)}})

print(json.dumps(results))
'''

        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{self.model_path}:/app/model:ro",
            "-v", f"{font_dir}:/fonts:ro",
            "-v", f"{Path.cwd()}:/app/workspace:ro",
            self.IMAGE, "python3", "-c", code
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"InkSight Docker failed: {result.stderr}")

        output_lines = result.stdout.strip().split('\n')
        return json.loads(output_lines[-1])


def test():
    """Test the Docker wrapper."""
    print("Testing InkSight Docker wrapper...")

    inksight = InkSightDocker()

    # Test single word
    result = inksight.process(
        '/home/server/glossy/font_scraper/fonts/dafont/Braydon Script.ttf',
        'hello',
        output_image='/tmp/inksight_docker_test.png'
    )

    print(f"Word: {result['word']}")
    print(f"Strokes: {result['num_strokes']}")
    print(f"Total points: {result['total_points']}")
    print(f"Visualization: {result.get('visualization', 'N/A')}")


if __name__ == '__main__':
    test()
