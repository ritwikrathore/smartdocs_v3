"""Quick test script to verify pycache cleanup functionality."""
import os
import shutil
import logging
from pathlib import Path

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def cleanup_pycache_test():
    """Test version of cleanup_pycache function."""
    cache_dirs_removed = 0
    cache_files_removed = 0
    
    try:
        # Start from the project root
        project_root = Path(__file__).parent
        
        # Directories to exclude from cleanup
        exclude_dirs = {'.venv', 'venv', 'env', 'node_modules', '.git', 'site-packages'}
        
        # Find and remove all __pycache__ directories
        for pycache_dir in project_root.rglob("__pycache__"):
            try:
                # Skip if any parent directory is in exclude list
                if any(excluded in pycache_dir.parts for excluded in exclude_dirs):
                    logger.info(f"Skipping excluded directory: {pycache_dir}")
                    continue
                    
                if pycache_dir.is_dir():
                    # Count files before removal
                    file_count = len(list(pycache_dir.iterdir()))
                    shutil.rmtree(pycache_dir, ignore_errors=True)
                    cache_dirs_removed += 1
                    cache_files_removed += file_count
                    logger.info(f"Removed __pycache__ directory: {pycache_dir}")
            except Exception as e:
                logger.error(f"Error removing __pycache__ directory {pycache_dir}: {str(e)}")
        
        # Also remove .pyc files that might exist outside __pycache__
        for pyc_file in project_root.rglob("*.pyc"):
            try:
                # Skip if any parent directory is in exclude list
                if any(excluded in pyc_file.parts for excluded in exclude_dirs):
                    continue
                    
                if pyc_file.is_file():
                    os.remove(pyc_file)
                    cache_files_removed += 1
                    logger.info(f"Removed .pyc file: {pyc_file}")
            except Exception as e:
                logger.error(f"Error removing .pyc file {pyc_file}: {str(e)}")
        
        logger.info(f"\n✅ Cleaned up {cache_dirs_removed} __pycache__ directories and {cache_files_removed} cache files")
    except Exception as e:
        logger.error(f"Error during Python cache cleanup: {str(e)}")

if __name__ == "__main__":
    print("Testing cleanup_pycache function...")
    cleanup_pycache_test()
    print("Test completed!")

