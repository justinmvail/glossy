#!/usr/bin/env python3
"""DiffVG Docker Wrapper for differentiable stroke optimization.

This module provides a Python interface to run DiffVG-based stroke optimization
inside a Docker container with GPU support. DiffVG is a differentiable vector
graphics rasterizer that enables gradient-based optimization of polyline strokes
against font glyph masks.

The Docker isolation ensures compatibility with CUDA and avoids dependency
conflicts between TensorFlow (used by InkSight) and PyTorch (used by DiffVG).

Docker Image Requirements:
    The container must have:
    - NVIDIA GPU support via nvidia-docker
    - Python 3.8+ with PyTorch and pydiffvg
    - The optimize_strokes.py script at /app/optimize_strokes.py

Volume Mounts:
    - Font directory mounted at /fonts (read-only)
    - Input JSON config mounted at /app/input.json (read-only)

Example:
    Basic usage to optimize strokes for a character::

        from docker.diffvg_docker import DiffVGDocker

        optimizer = DiffVGDocker()
        result = optimizer.optimize(
            font_path='fonts/dafont/MyFont.ttf',
            char='B',
            initial_strokes=[[[10, 20], [30, 40], [50, 60]]],
        )

        if 'error' not in result:
            print(f"Score: {result['score']}, Strokes: {len(result['strokes'])}")
        else:
            print(f"Optimization failed: {result['error']}")

Attributes:
    IMAGE (str): Docker image name for the DiffVG optimizer container.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional


class DiffVGDocker:
    """Docker wrapper for running DiffVG stroke optimization.

    This class manages the lifecycle of Docker containers that run differentiable
    stroke optimization using pydiffvg. It handles GPU detection, fallback to CPU,
    input/output serialization, and container execution.

    The optimization process takes initial polyline strokes and refines them using
    gradient descent to better match a target font glyph mask. The differentiable
    rasterizer enables computing gradients through the rendering process.

    Attributes:
        IMAGE (str): Docker image name ('diffvg-optimizer:latest').
        use_gpu (bool): Whether to attempt GPU execution.

    Example:
        >>> optimizer = DiffVGDocker(use_gpu=True)
        >>> result = optimizer.optimize(
        ...     font_path='/path/to/font.ttf',
        ...     char='A',
        ...     initial_strokes=[[[10, 20], [30, 40], [50, 60]]],
        ...     num_iterations=500
        ... )
        >>> print(f"Final score: {result.get('score', 'N/A')}")
    """

    IMAGE = "diffvg-optimizer:latest"

    def __init__(self, use_gpu: bool = True):
        """Initialize the DiffVG Docker wrapper.

        Args:
            use_gpu: If True, attempts to run the container with GPU support
                (--gpus all). If GPU execution fails (e.g., no NVIDIA driver),
                the wrapper automatically falls back to CPU mode on subsequent
                calls.
        """
        self.use_gpu = use_gpu
        self._docker_available = None

    def _check_docker(self) -> bool:
        """Verify that Docker is available on the system.

        Caches the result after the first check to avoid repeated subprocess
        calls.

        Returns:
            True if Docker CLI is available and functional, False otherwise.
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

        When the current user is not in the docker group, commands must be
        wrapped with 'sg docker -c' to execute with docker group privileges.
        This is common in multi-user systems where docker access is controlled
        via group membership.

        Args:
            cmd: List of command arguments to potentially wrap.

        Returns:
            The original command if the user is in the docker group, or the
            command wrapped with 'sg docker -c' otherwise.
        """
        import grp, os, shlex
        try:
            docker_gid = grp.getgrnam('docker').gr_gid
            if docker_gid in os.getgroups():
                return cmd
        except KeyError:
            pass
        # Use sg to run with docker group privileges
        return ["sg", "docker", "-c", " ".join(shlex.quote(c) for c in cmd)]

    def optimize(
        self,
        font_path: str,
        char: str,
        initial_strokes: Optional[List] = None,
        canvas_size: int = 224,
        num_iterations: int = 500,
        stroke_width: float = 8.0,
        thin_iterations: int = 0,
        timeout: int = 300,
    ) -> Dict:
        """Optimize strokes to match a font glyph using DiffVG.

        Runs the differentiable stroke optimization inside a Docker container.
        The optimizer uses gradient descent (Adam) to adjust stroke point
        positions and widths to minimize the difference between rendered
        strokes and the target glyph mask.

        The optimization includes several loss components:
        - Coverage: Penalizes uncovered glyph pixels
        - Outside penalty: Penalizes strokes outside the glyph
        - Smoothness: Encourages smooth stroke curves
        - Anchor: Preserves overall stroke topology

        Args:
            font_path: Absolute or relative path to the font file (.ttf, .otf).
                The file must be accessible from the host system.
            char: Single character to optimize strokes for.
            initial_strokes: List of strokes as starting points for optimization.
                Each stroke is a list of [x, y] coordinate pairs:
                [[[x1, y1], [x2, y2], ...], ...]. If None, optimization will
                fail as initial strokes are required.
            canvas_size: Size of the square canvas in pixels. Default is 224
                to match InkSight's coordinate space.
            num_iterations: Maximum number of Adam optimizer iterations.
                More iterations may improve results but increase runtime.
            stroke_width: Initial stroke width in pixels. The optimizer may
                adjust this during optimization (range: 2.0-25.0).
            thin_iterations: Number of topology-preserving thinning iterations
                to apply to the target glyph mask. Reduces junction blobs in
                thick fonts. Default 0 (no thinning).
            timeout: Maximum seconds to wait for the container to complete.
                Default is 300 (5 minutes).

        Returns:
            A dictionary with optimization results:
            - 'strokes': List of optimized strokes as [[[x, y], ...], ...]
            - 'score': Coverage score (0-1, higher is better)
            - 'elapsed': Optimization time in seconds
            - 'iterations': Number of iterations configured
            - 'final_loss': Final loss value from optimization

            On failure, returns {'error': str} with an error message.

        Example:
            >>> result = optimizer.optimize(
            ...     font_path='fonts/Arial.ttf',
            ...     char='A',
            ...     initial_strokes=[
            ...         [[50, 200], [112, 20], [174, 200]],  # Left diagonal
            ...         [[80, 130], [144, 130]],  # Crossbar
            ...     ],
            ...     num_iterations=300,
            ... )
            >>> if 'error' not in result:
            ...     print(f"Score: {result['score']:.3f}")
        """
        if not self._check_docker():
            return {'error': 'Docker not available'}

        font_path = str(Path(font_path).resolve())
        font_dir = str(Path(font_path).parent)
        font_name = Path(font_path).name

        # Build input JSON
        config = {
            'font_path': f'/fonts/{font_name}',
            'char': char,
            'canvas_size': canvas_size,
            'num_iterations': num_iterations,
            'stroke_width': stroke_width,
            'thin_iterations': thin_iterations,
        }
        if initial_strokes:
            config['initial_strokes'] = initial_strokes

        # Write config to temp file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False, dir='/tmp'
        ) as f:
            json.dump(config, f)
            config_path = f.name

        try:
            # Build docker command
            cmd = ["docker", "run", "--rm"]

            if self.use_gpu:
                cmd.extend(["--gpus", "all"])

            cmd.extend([
                "-v", f"{font_dir}:/fonts:ro",
                "-v", f"{config_path}:/app/input.json:ro",
                self.IMAGE,
                "python3", "/app/optimize_strokes.py", "/app/input.json",
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
                    return self.optimize(
                        font_path, char, initial_strokes,
                        canvas_size, num_iterations, stroke_width, timeout,
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
