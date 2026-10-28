import cv2
import numpy as np
import os

# Create a folder for our experiment results.
os.makedirs("results", exist_ok=True)


# Calculate the average squared pixel difference.
def calculate_mse(original, comparison):
    difference = (
        original.astype(np.float32)
        - comparison.astype(np.float32)
    )
    return np.mean(difference ** 2)


# Reconstruct an image using Richardson-Lucy deconvolution.
def reconstruct_image(blurred, kernel_size, iterations=20):

    # Convert pixel values from 0-255 to 0-1.
    observed = blurred.astype(np.float32) / 255.0

    # Create the assumed Gaussian blur kernel.
    one_d_kernel = cv2.getGaussianKernel(kernel_size, 0)
    kernel = one_d_kernel @ one_d_kernel.T

    # Start with the blurred image as our estimate.
    estimate = observed.copy()

    # Repeatedly improve the estimate.
    for _ in range(iterations):

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

    # Convert the result back to image pixels.
    return (estimate * 255).astype(np.uint8)


# Process our three test images.
for number in range(1, 4):

    filename = f"test{number}.png"
    original = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Stop if an image cannot be found.
    if original is None:
        print(f"Could not load {filename}. Check the filename.")
        continue

    # Create the same controlled blur for each image.
    blurred = cv2.GaussianBlur(original, (31, 31), 0)

    # Reconstruct using the correct and incorrect kernel sizes.
    baseline = reconstruct_image(blurred, kernel_size=31)
    wrong_model = reconstruct_image(blurred, kernel_size=15)

    # Save all results separately.
    cv2.imwrite(f"results/test{number}_blurred.png", blurred)
    cv2.imwrite(f"results/test{number}_baseline.png", baseline)
    cv2.imwrite(f"results/test{number}_wrong_kernel.png", wrong_model)

    # Calculate pixel-wise errors against the original.
    blurred_mse = calculate_mse(original, blurred)
    baseline_mse = calculate_mse(original, baseline)
    wrong_mse = calculate_mse(original, wrong_model)

    print(f"\n--- {filename} ---")
    print(f"Blurred MSE:              {blurred_mse:.2f}")
    print(f"Correct-model MSE:        {baseline_mse:.2f}")
    print(f"Wrong-kernel MSE:         {wrong_mse:.2f}")

print("\nDataset evaluation complete!")
