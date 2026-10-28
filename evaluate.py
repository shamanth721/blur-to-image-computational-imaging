import cv2
import numpy as np
import math

# Load all three images in grayscale.
original = cv2.imread("original.png", cv2.IMREAD_GRAYSCALE)
blurred = cv2.imread("blurred.png", cv2.IMREAD_GRAYSCALE)
reconstructed = cv2.imread("reconstructed.png", cv2.IMREAD_GRAYSCALE)

# Check that all images were loaded.
if original is None or blurred is None or reconstructed is None:
    print("Error: An image could not be loaded.")
    raise SystemExit

# Calculate Mean Squared Error.
def calculate_mse(reference, image):
    difference = reference.astype(np.float32) - image.astype(np.float32)
    return np.mean(difference ** 2)

# Calculate Peak Signal-to-Noise Ratio.
def calculate_psnr(reference, image):
    mse = calculate_mse(reference, image)

    if mse == 0:
        return float("inf")

    return 10 * math.log10((255 ** 2) / mse)

# Compare each result with the original reference.
for name, image in [
    ("Blurred", blurred),
    ("Reconstructed", reconstructed)
]:
    mse = calculate_mse(original, image)
    psnr = calculate_psnr(original, image)

    print(f"\n{name} image:")
    print(f"MSE:  {mse:.2f}")
    print(f"PSNR: {psnr:.2f} dB")