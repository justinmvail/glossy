#!/usr/bin/env python3
"""Stroke Editor - Web app for viewing and editing InkSight stroke data.

This is the main entry point for the stroke editor Flask application.
Route handlers are split across stroke_routes_core.py and stroke_routes_batch.py.
"""

# Import Flask app and register routes
from stroke_flask import app
import stroke_routes_core   # noqa: F401 - registers core routes with app
import stroke_routes_batch  # noqa: F401 - registers batch routes with app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
