import os
import sys

# Ensure root directory is in path for clean imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ui.gradio_app import create_gradio_app

if __name__ == "__main__":
    demo = create_gradio_app()
    
    # RAILWAY / RENDER COMPATIBILITY:
    # These platforms set a PORT environment variable dynamically.
    # Default to 7860 for local testing if not set.
    port = int(os.environ.get("PORT", 7860))
    
    print(f"🚀 Starting server on port {port}...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )
