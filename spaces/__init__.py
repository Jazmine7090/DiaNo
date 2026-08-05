import os
import sys

# Check if we are running in the Hugging Face Spaces environment
if "SPACE_ID" in os.environ:
    # Temporarily remove the local workspace directory from sys.path
    # so we load Hugging Face's system-installed 'spaces' package
    current_dir = os.path.dirname(os.path.dirname(__file__))
    saved_path = list(sys.path)
    if current_dir in sys.path:
        sys.path.remove(current_dir)
    try:
        import spaces as hf_spaces
        GPU = hf_spaces.GPU
    except Exception:
        # Fallback
        def GPU(func):
            return func
    finally:
        sys.path = saved_path
else:
    # Local fallback decorator
    def GPU(func):
        return func
