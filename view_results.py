import cv2
import matplotlib.pyplot as plt

# Create a figure with 3 rows and 4 columns.
fig, axes = plt.subplots(3, 4, figsize=(15, 10))

for number in range(1, 4):

    # Load the four versions of each test image.
    images = [
        cv2.imread(f"test{number}.png", cv2.IMREAD_GRAYSCALE),
        cv2.imread(
            f"results/test{number}_blurred.png",
            cv2.IMREAD_GRAYSCALE
        ),
        cv2.imread(
            f"results/test{number}_baseline.png",
            cv2.IMREAD_GRAYSCALE
        ),
        cv2.imread(
            f"results/test{number}_wrong_kernel.png",
            cv2.IMREAD_GRAYSCALE
        ),
    ]

    titles = [
        "Original",
        "Blurred",
        "Correct model",
        "Wrong kernel"
    ]

    for column, image in enumerate(images):

        if image is None:
            print(f"Missing image for test {number}.")
            raise SystemExit

        axes[number - 1, column].imshow(
            image, cmap="gray", vmin=0, vmax=255
        )
        axes[number - 1, column].set_title(titles[column])
        axes[number - 1, column].axis("off")

    axes[number - 1, 0].set_ylabel(
        f"Test {number}", fontsize=11
    )

plt.tight_layout()
plt.show()