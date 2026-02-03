#!/usr/bin/env python3
"""
DiffVG Docker Wrapper

Run differentiable stroke optimization in a Docker container with GPU support.
Uses pydiffvg for gradient-based optimization of polyline strokes against font
glyph masks.

Usage:
    from docker.diffvg_docker import DiffVGDocker

    optimizer = DiffVGDocker()
    result = optimizer.optimize(
        font_path='fonts/dafont/MyFont.ttf',
        char='B',
        initial_strokes=[[[10, 20], [30, 40], ...]],
    )
    print(f"Score: {result['score']}, Strokes: {len(result['strokes'])}")
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional


class DiffVGDocker:
    """Run DiffVG stroke optimization in a Docker container."""

    IMAGE = "diffvg-optimizer:latest"

    def __init__(self, use_gpu: bool = True):
        """
        Args:
            use_gpu: Try to use GPU (--gpus all). Falls back to CPU on failure.
        """
        self.use_gpu = use_gpu
        self._docker_available = None

    def _check_docker(self) -> bool:
        """Verify Docker is available."""
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
        """Wrap command with sg docker if docker group not in current groups."""
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
        timeout: int = 300,
    ) -> Dict:
        """
        Optimize strokes to match a font glyph using DiffVG.

        Args:
            font_path: Path to font file (.ttf, .otf)
            char: Character to optimize
            initial_strokes: List of strokes [[[x,y], ...], ...] as starting point
            canvas_size: Canvas size in pixels (default 224)
            num_iterations: Number of Adam optimizer iterations
            stroke_width: Initial stroke width in pixels
            timeout: Max seconds to wait for container

        Returns:
            Dict with 'strokes', 'score', 'elapsed', 'iterations', 'final_loss'.
            Returns {'error': str} on failure.
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
