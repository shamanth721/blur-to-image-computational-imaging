# Check that Python and our libraries are working.

import sys
import cv2
import numpy as np
import matplotlib

print("Python version:", sys.version.split()[0])
print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
print("Matplotlib version:", matplotlib.__version__)

print("\nSetup successful!")