import sys
import importlib
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

importlib.invalidate_caches()
try:
    # Import SmartReview from src/smartreview directory
    import src.smartreview.smartreview as SR
    print('Imported SmartReview OK')
except Exception:
    traceback.print_exc()
    print('Import failed')
