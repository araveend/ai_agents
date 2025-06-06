import shutil
import os

def clean_pycache(root_dir='.'):
    """Recursively removes __pycache__ directories and .pyc files."""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '__pycache__' in dirnames:
            shutil.rmtree(os.path.join(dirpath, '__pycache__'))
            print(f"Removed: {os.path.join(dirpath, '__pycache__')}")
        for filename in filenames:
            if filename.endswith('.pyc'):
                os.remove(os.path.join(dirpath, filename))
                print(f"Removed: {os.path.join(dirpath, filename)}")

if __name__ == "__main__":
    clean_pycache()
    print("Cleanup complete.")