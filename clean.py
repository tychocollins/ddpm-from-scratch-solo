import os
import glob

# This finds and deletes all .pt files in the current folder
files = glob.glob("*.pt")
for f in files:
    try:
        os.remove(f)
        print(f"Successfully deleted: {f}")
    except OSError as e:
        print(f"Error deleting {f}: {e}")

print("Clean-up complete. You are ready for a fresh start!")
