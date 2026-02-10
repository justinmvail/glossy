#!/usr/bin/env python3
"""InkSight Docker Wrapper for font vectorization.

This module provides a Python interface to run Google's InkSight model inside
a Docker container with GPU support. InkSight is a Vision-Language model that
converts images of handwriting to digital ink (vector strokes).

Docker isolation is essential because InkSight requires TensorFlow, while other
components in the font_scraper pipeline use PyTorch. Running them in separate
containers avoids GPU memory conflicts and dependency version issues.

Docker Image Requirements:
    The inksight Docker image must include:
    - Python 3.8+ with TensorFlow 2.15-2.17
    - tensorflow-text package (required by InkSight model)
    - The InkSight saved_model directory mounted at /app/model
    - Access to the inksight_vectorizer.py module

Volume Mounts:
    - Model directory: /app/model (read-only)
    - Font directory: /fonts (read-only)
    - Workspace: /app/workspace (for accessing vectorizer code, read-only)
    - Output directory: mounted if saving visualizations

Example:
    Basic usage to vectorize a word::

        from docker.inksight_docker import InkSightDocker

        inksight = InkSightDocker()
        result = inksight.process(
            font_path='fonts/dafont/MyFont.ttf',
            word='hello',
            output_image='/tmp/visualization.png'
        )
        print(f"Strokes: {result['num_strokes']}")
        print(f"Total points: {result['total_points']}")

    Batch processing multiple words::

        results = inksight.batch_process(
            font_path='fonts/MyFont.ttf',
            words=['hello', 'world', 'test']
        )
        for r in results:
            if r['success']:
                print(f"{r['word']}: {r['num_strokes']} strokes")

Attributes:
    IMAGE (str): Default Docker image name ('inksight:latest').
    MODEL_PATH (str): Default path to InkSight model on host.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import base64


class InkSightDocker:
    """Docker wrapper for running InkSight font vectorization.

    This class manages Docker container execution for InkSight inference,
    handling model mounting, font access, and result parsing. It provides
    both single-word and batch processing interfaces.

    InkSight converts rendered font images to vector strokes by treating
    them as handwriting samples. This produces natural stroke orderings
    that follow how a human would write the character.

    Attributes:
        IMAGE (str): Docker image name ('inksight:latest').
        MODEL_PATH (str): Default host path to the InkSight model directory.
        model_path (str): Configured model path for this instance.

    Example:
        >>> inksight = InkSightDocker(model_path='/custom/model/path')
        >>> result = inksight.process('/fonts/Arial.ttf', 'Hello')
        >>> print(f"Generated {result['num_strokes']} strokes")
    """

    IMAGE = "inksight:latest"
    MODEL_PATH = "/home/server/inksight/model"

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the InkSight Docker wrapper.

        Verifies that Docker is available on the system. Raises an error
        immediately if Docker is not installed or not running, since all
        operations require Docker.

        Args:
            model_path: Path to the InkSight saved_model directory on the host
                system. This directory will be mounted into the container at
                /app/model. If None, uses the default MODEL_PATH.

        Raises:
            RuntimeError: If Docker is not installed or not running.

        Example:
            >>> inksight = InkSightDocker()  # Uses default model path
            >>> inksight = InkSightDocker('/custom/path/to/model')
        """
        self.model_path = model_path or self.MODEL_PATH
        self._check_docker()

    def _check_docker(self):
        """Verify that Docker is available and functional.

        Runs 'docker --version' to check Docker availability. This is called
        during initialization to fail fast if Docker is not available.

        Raises:
            RuntimeError: If Docker CLI is not found or returns an error.
        """
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
        """Vectorize a word using InkSight in Docker.

        Renders the specified word using the given font, runs InkSight
        inference to extract vector strokes, and returns the results.
        Optionally saves a visualization showing the original image
        alongside the extracted strokes.

        The process involves:
        1. Starting a Docker container with GPU support
        2. Loading the InkSight model
        3. Rendering the word with the font
        4. Running InkSight inference to get stroke tokens
        5. Post-processing: filtering artifacts, smoothing, connecting gaps
        6. Returning stroke coordinates

        Args:
            font_path: Path to the font file (.ttf, .otf, .woff2) on the host
                system. The parent directory is mounted into the container.
            word: The word or text to render and vectorize. Can be a single
                character or a full word.
            output_image: Optional path to save a visualization PNG showing
                the input image and extracted strokes side by side. If None,
                no visualization is saved.

        Returns:
            A dictionary containing:
            - 'strokes': List of stroke point arrays, each as [[x, y], ...]
            - 'num_strokes': Number of strokes extracted
            - 'total_points': Total number of points across all strokes
            - 'word': The input word that was processed
            - 'visualization': Path to saved image (only if output_image was set)

        Raises:
            RuntimeError: If the Docker container fails to execute or returns
                a non-zero exit code.

        Example:
            >>> result = inksight.process(
            ...     '/fonts/Script.ttf',
            ...     'Hello',
            ...     output_image='/tmp/hello_strokes.png'
            ... )
            >>> print(f"Extracted {result['num_strokes']} strokes")
            >>> for i, stroke in enumerate(result['strokes']):
            ...     print(f"Stroke {i}: {len(stroke)} points")
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
        """Process multiple words in a single container run.

        More efficient than calling process() multiple times because the model
        is loaded only once and the container overhead is amortized across all
        words. Recommended when processing many words from the same font.

        Each word is processed independently, and failures for individual words
        do not affect the processing of other words in the batch.

        Args:
            font_path: Path to the font file (.ttf, .otf, .woff2) on the host.
            words: List of words to process. Each word is vectorized separately.

        Returns:
            A list of dictionaries, one per input word, in the same order:
            - On success:
                - 'word': The processed word
                - 'strokes': List of stroke point arrays
                - 'num_strokes': Number of strokes
                - 'total_points': Total point count
                - 'success': True
            - On failure:
                - 'word': The word that failed
                - 'success': False
                - 'error': Error message string

        Raises:
            RuntimeError: If the Docker container itself fails to start or
                crashes. Individual word failures are captured in the results.

        Example:
            >>> results = inksight.batch_process(
            ...     '/fonts/Script.ttf',
            ...     ['Hello', 'World', 'Test']
            ... )
            >>> for r in results:
            ...     if r['success']:
            ...         print(f"{r['word']}: {r['num_strokes']} strokes")
            ...     else:
            ...         print(f"{r['word']}: FAILED - {r['error']}")
        """
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
    """Test the InkSight Docker wrapper with a sample font.

    Runs a quick integration test that:
    1. Creates an InkSightDocker instance
    2. Processes a single word from a test font
    3. Saves a visualization to /tmp
    4. Prints results summary

    This function is useful for verifying that Docker, GPU, and the InkSight
    model are all configured correctly.

    Example:
        >>> test()
        Testing InkSight Docker wrapper...
        Word: hello
        Strokes: 5
        Total points: 127
        Visualization: /tmp/inksight_docker_test.png
    """
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
