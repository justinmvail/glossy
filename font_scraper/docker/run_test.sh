#!/bin/bash
docker run --rm --gpus all -v /home/server/inksight/model:/app/model:ro -v /home/server/glossy/font_scraper:/app/workspace:ro -v /tmp:/tmp inksight:latest python3 /app/workspace/docker/test_inksight.py
