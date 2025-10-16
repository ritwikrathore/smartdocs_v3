import sys
sys.path.append(r'c:\Users\rrathore1\Projects\keyword_code_v3')
import importlib
import traceback

importlib.invalidate_caches()
try:
    import SmartReview as SR
    print('Imported SmartReview OK')
except Exception:
    traceback.print_exc()
    print('Import failed')
