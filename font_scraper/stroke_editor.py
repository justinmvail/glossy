#!/usr/bin/env python3
"""Stroke Editor - Web application for viewing and editing InkSight stroke data.

This module serves as the main entry point for the Stroke Editor Flask application.
It imports and registers all route handlers from the modular route files, then
starts the Flask development server when run directly.

The Stroke Editor provides a web-based interface for:
    - Browsing fonts in the database
    - Viewing character glyphs rendered from font files
    - Creating and editing stroke data for characters
    - Processing strokes with smoothing and connection algorithms
    - Running batch operations on fonts
    - Real-time progress streaming via Server-Sent Events (SSE)

Application Architecture:
    The application follows a modular design pattern where routes are organized
    into separate modules by functionality:

    - stroke_flask.py: Core Flask app instance, database utilities, and constants
    - stroke_routes_core.py: Character editing, rendering, and font management
    - stroke_routes_batch.py: Batch processing operations for multiple fonts
    - stroke_routes_stream.py: SSE endpoints for streaming progress updates

    Importing the route modules automatically registers their routes with the
    Flask application instance due to the use of the @app.route decorator.

Usage:
    Run directly to start the development server::

        python stroke_editor.py

    Or import for use with a production WSGI server::

        from stroke_editor import app
        # Use with gunicorn, uwsgi, etc.

Configuration:
    The server runs with the following default settings:
        - Debug mode: enabled
        - Host: 0.0.0.0 (accessible from all network interfaces)
        - Port: 5000

    For production deployments, use a proper WSGI server and disable debug mode.

Attributes:
    app (Flask): The Flask application instance, imported from stroke_flask.
        This is the WSGI application object that can be used with production
        servers like gunicorn or uwsgi.

Example:
    Starting the development server::

        $ python stroke_editor.py
        * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
        * Restarting with stat
        * Debugger is active!

    Using with gunicorn::

        $ gunicorn -w 4 -b 0.0.0.0:5000 stroke_editor:app
"""

# Import Flask app and register routes
import os

import stroke_routes_batch  # noqa: F401 - registers batch routes with app
import stroke_routes_core  # noqa: F401 - registers core routes with app
import stroke_routes_stream  # noqa: F401 - registers SSE streaming routes with app
from stroke_flask import app

if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', '1').lower() in ('1', 'true', 'yes')
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', '5000'))
    app.run(debug=debug, host=host, port=port)
