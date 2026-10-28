import cv2
import matplotlib.pyplot as plt

# Step 1: Load the original image.
original = cv2.imread("original.png")

# Check whether the image was found.
if original is None:
    print("Error: Could not find original.png")
    raise SystemExit

# Step 2: Apply Gaussian blur.
# The 31 x 31 setting controls the strength of the blur.
blurred = cv2.GaussianBlur(original, (31, 31), 0)

# Step 3: Save the blurred image.
cv2.imwrite("blurred.png", blurred)

# Step 4: Convert image colours for correct display in Matplotlib.
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

# Step 5: Display both images side by side.
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(blurred_rgb)
plt.title("Blurred Image")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Experiment complete!")
print("Blurred image saved as blurred.png")