import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the original, blurred, and baseline reconstructed images.
original = cv2.imread("original.png", cv2.IMREAD_GRAYSCALE)
blurred = cv2.imread("blurred.png", cv2.IMREAD_GRAYSCALE)
baseline = cv2.imread("reconstructed.png", cv2.IMREAD_GRAYSCALE)

# Stop if any required image is missing.
if original is None or blurred is None or baseline is None:
    print("Error: Could not load one or more images.")
    raise SystemExit

# Convert the blurred image to numbers between 0 and 1.
observed = blurred.astype(np.float32) / 255.0

# Experiment: assume the blur kernel was 15 x 15, not 31 x 31.
one_d_kernel = cv2.getGaussianKernel(15, 0)
kernel = one_d_kernel @ one_d_kernel.T

# Start with the observed blurred image as our estimate.
estimate = observed.copy()

# Apply Richardson-Lucy reconstruction for 20 iterations.
for i in range(20):
    predicted_blur = cv2.filter2D(
        estimate,
        -1,
        kernel,
        borderType=cv2.BORDER_REFLECT_101
    )

    ratio = observed / (predicted_blur + 1e-7)

    correction = cv2.filter2D(
        ratio,
        -1,
        kernel,
        borderType=cv2.BORDER_REFLECT_101
    )

    estimate = estimate * correction
    estimate = np.clip(estimate, 0, 1)

# Convert the result back to an image.
wrong_model_result = (estimate * 255).astype(np.uint8)

# Save the new result without overwriting the baseline.
cv2.imwrite("reconstructed_kernel15.png", wrong_model_result)

# Calculate Mean Squared Error against the original.
def calculate_mse(reference, image):
    difference = (
        reference.astype(np.float32)
        - image.astype(np.float32)
    )
    return np.mean(difference ** 2)

print("Baseline reconstruction MSE:", round(calculate_mse(original, baseline), 2))
print("Wrong-kernel reconstruction MSE:", round(calculate_mse(original, wrong_model_result), 2))

# Compare the images visually.
plt.figure(figsize=(16, 5))

images = [
    (original, "Original"),
    (blurred, "Blurred input"),
    (baseline, "Correct-model baseline"),
    (wrong_model_result, "Wrong-model reconstruction")
]

for i, (image, title) in enumerate(images, start=1):
    plt.subplot(1, 4, i)
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()

print("Experiment complete!")
print("Saved as reconstructed_kernel15.png")