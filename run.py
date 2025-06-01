# run.py
#!/usr/bin/env python
import os
import sys
import argparse
from app.gradio_app import create_and_launch_app
from app.config import DEFAULT_CONFIG

def main():
    parser = argparse.ArgumentParser(description="NeuroNest - OneFormer with Contrast Detection")
    parser.add_argument("--port", type=int, default=7860, help="Port to run Gradio app on")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = DEFAULT_CONFIG.copy()
    config.update({
        "port": args.port,
        "share": args.share,
        "debug": args.debug
    })
    
    # Launch the app
    create_and_launch_app(config)

if __name__ == "__main__":
    main()
