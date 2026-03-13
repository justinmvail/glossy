"""Stroke Model Docker Wrapper for ML-based stroke prediction.

This module provides a Python interface to run the trained stroke prediction
model inside a Docker container with GPU support. Follows the same patterns
as diffvg_docker.py.

Volume Mounts:
    - Font directory mounted at /fonts (read-only)
    - Model checkpoint directory mounted at /app/checkpoints (read-only)
    - Input JSON config mounted at /app/input.json (read-only)

Example:
    Basic usage to predict strokes for a character::

        from docker.stroke_model_docker import StrokeModelDocker

        predictor = StrokeModelDocker()
        result = predictor.predict(
            font_path='fonts/dafont/MyFont.ttf',
            char='B',
        )

        if 'error' not in result:
            print(f"Score: {result['score']}, Strokes: {len(result['strokes'])}")
        else:
            print(f"Prediction failed: {result['error']}")

Attributes:
    IMAGE (str): Docker image name for the stroke model container.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict


class StrokeModelDocker:
    """Docker wrapper for running ML stroke prediction.

    This class manages the lifecycle of Docker containers that run the
    trained CNN+Transformer stroke prediction model. It handles GPU
    detection, fallback to CPU, input/output serialization, and container
    execution.

    Attributes:
        IMAGE (str): Docker image name ('stroke-model:latest').
        use_gpu (bool): Whether to attempt GPU execution.
        checkpoint_dir (str): Path to directory containing model checkpoints.
    """

    IMAGE = "stroke-model:latest"

    def __init__(self, use_gpu: bool = True, checkpoint_dir: str = None):
        """Initialize the Stroke Model Docker wrapper.

        Args:
            use_gpu: If True, attempts to run the container with GPU support
                (--gpus all). Falls back to CPU on GPU failure.
            checkpoint_dir: Path to directory containing model checkpoints.
                Defaults to docker/stroke_model/checkpoints/ relative to
                this file's location.
        """
        self.use_gpu = use_gpu
        self._docker_available = None

        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'stroke_model', 'checkpoints',
            )
        self.checkpoint_dir = checkpoint_dir

    def _check_docker(self) -> bool:
        """Verify that Docker is available on the system.

        Caches the result after the first check.

        Returns:
            True if Docker CLI is available, False otherwise.
        """
        if self._docker_available is not None:
            return self._docker_available
        try:
            subprocess.run(
                self._wrap_cmd(["docker", "--version"]),
                capture_output=True, text=True, check=True,
            )
            self._docker_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self._docker_available = False
        return self._docker_available

    @staticmethod
    def _wrap_cmd(cmd):
        """Wrap a command with 'sg docker' if needed for group privileges.

        Args:
            cmd: List of command arguments to potentially wrap.

        Returns:
            The original command or wrapped with 'sg docker -c'.
        """
        import grp, shlex
        try:
            docker_gid = grp.getgrnam('docker').gr_gid
            if docker_gid in os.getgroups():
                return cmd
        except KeyError:
            pass
        return ["sg", "docker", "-c", " ".join(shlex.quote(c) for c in cmd)]

    def predict(
        self,
        font_path: str,
        char: str,
        canvas_size: int = 224,
        existence_threshold: float = 0.5,
        timeout: int = 120,
    ) -> Dict:
        """Predict strokes for a font character using the trained model.

        Runs inference inside a Docker container.

        Args:
            font_path: Path to the font file (.ttf, .otf).
            char: Single character to predict strokes for.
            canvas_size: Size of the square canvas in pixels.
            existence_threshold: Minimum probability to include a stroke.
            timeout: Maximum seconds to wait for the container.

        Returns:
            Dict with prediction results:
            - 'strokes': List of strokes as [[[x, y], ...], ...]
            - 'score': Coverage score (0-1)
            - 'n_strokes': Number of predicted strokes
            - 'elapsed': Inference time in seconds

            On failure, returns {'error': str}.
        """
        if not self._check_docker():
            return {'error': 'Docker not available'}

        font_path = str(Path(font_path).resolve())
        font_dir = str(Path(font_path).parent)
        font_name = Path(font_path).name

        config = {
            'font_path': f'/fonts/{font_name}',
            'char': char,
            'canvas_size': canvas_size,
            'existence_threshold': existence_threshold,
            'model_path': '/app/checkpoints/best_model.pt',
        }

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False, dir='/tmp'
        ) as f:
            json.dump(config, f)
            config_path = f.name

        try:
            cmd = ["docker", "run", "--rm"]

            if self.use_gpu:
                cmd.extend(["--gpus", "all"])

            cmd.extend([
                "-v", f"{font_dir}:/fonts:ro",
                "-v", f"{self.checkpoint_dir}:/app/checkpoints:ro",
                "-v", f"{config_path}:/app/input.json:ro",
                self.IMAGE,
                "python3", "/app/predict.py", "/app/input.json",
            ])

            result = subprocess.run(
                self._wrap_cmd(cmd),
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                # If GPU failed, retry without GPU
                if self.use_gpu and ('nvidia' in stderr.lower() or
                                     'gpu' in stderr.lower() or
                                     'cuda' in stderr.lower()):
                    self.use_gpu = False
                    return self.predict(
                        font_path, char, canvas_size,
                        existence_threshold, timeout,
                    )
                return {'error': f'Container failed: {stderr[-500:]}'}

            # Parse JSON output (last line of stdout)
            output_lines = result.stdout.strip().split('\n')
            for line in reversed(output_lines):
                line = line.strip()
                if line.startswith('{'):
                    return json.loads(line)

            return {'error': 'No JSON output from container'}

        except subprocess.TimeoutExpired:
            return {'error': f'Container timed out after {timeout}s'}
        except json.JSONDecodeError as e:
            return {'error': f'Invalid JSON output: {e}'}
        finally:
            try:
                os.unlink(config_path)
            except OSError:
                pass
