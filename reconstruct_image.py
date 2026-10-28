import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the original and blurred images in grayscale.
original = cv2.imread("original.png", cv2.IMREAD_GRAYSCALE)
blurred = cv2.imread("blurred.png", cv2.IMREAD_GRAYSCALE)

# Step 2: Check that both images loaded successfully.
if original is None or blurred is None:
    print("Error: Could not load the images.")
    raise SystemExit

# Step 3: Convert pixel values from 0-255 to 0-1.
observed = blurred.astype(np.float32) / 255.0

# Step 4: Re-create the Gaussian blur kernel we used earlier.
one_d_kernel = cv2.getGaussianKernel(31, 0)
kernel = one_d_kernel @ one_d_kernel.T

# Step 5: Start with the blurred image as our first estimate.
estimate = observed.copy()

# Step 6: Improve the estimate over several iterations.
for i in range(20):

    # Predict what our current estimate looks like when blurred.
    predicted_blur = cv2.filter2D(
        estimate, -1, kernel,
        borderType=cv2.BORDER_REFLECT_101
    )

    # Compare the real blurred image with our prediction.
    ratio = observed / (predicted_blur + 1e-7)

    # Use that comparison to calculate a correction.
    correction = cv2.filter2D(
        ratio, -1, kernel,
        borderType=cv2.BORDER_REFLECT_101
    )

    # Update our estimate.
    estimate = estimate * correction

    # Keep pixel values within the valid range.
    estimate = np.clip(estimate, 0, 1)

# Step 7: Convert the reconstructed image back to 8-bit pixels.
reconstructed = (estimate * 255).astype(np.uint8)

# Step 8: Save the result.
cv2.imwrite("reconstructed.png", reconstructed)

# Step 9: Compare all three images.
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(blurred, cmap="gray", vmin=0, vmax=255)
plt.title("Blurred")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(reconstructed, cmap="gray", vmin=0, vmax=255)
plt.title("Reconstructed")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Reconstruction complete!")
print("Saved as reconstructed.png")