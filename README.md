# Blur-to-Image: A Computational Imaging Experiment

## Overview

This beginner computer vision project investigates how computational algorithms can reduce the effects of image blur.

Using Python and OpenCV, I simulate Gaussian blur on images, apply Richardson–Lucy deconvolution, and compare the reconstructed results with the original images.

The goal is not to claim perfect image recovery. Instead, the project investigates how reconstruction quality depends on the assumed blur model and evaluates the results using numerical measurements.

## Research Question

**How does a mismatch between the actual blur and the assumed blur kernel affect image reconstruction quality?**

## Methods

1. Capture or select an original image.
2. Convert it to grayscale for the initial experiment.
3. Apply a Gaussian blur using a 31 × 31 kernel.
4. Reconstruct the blurred image using Richardson–Lucy deconvolution.
5. Repeat reconstruction using an assumed 15 × 15 kernel.
6. Compare results against the original image using Mean Squared Error (MSE).

## Tools

- Python
- OpenCV
- NumPy
- Matplotlib

No deep-learning model was trained or used. The reconstruction algorithm is implemented using image-processing operations.

## Results

The following results were obtained from three test images using 20 reconstruction iterations.

| Test image                | Blurred MSE | Correct-model MSE | Wrong-kernel MSE |
| ------------------------- | ----------: | ----------------: | ---------------: |
| Test 1: Book cover        |      147.62 |            111.12 |           131.82 |
| Test 2: Portrait          |      222.29 |            141.31 |           190.67 |
| Test 3: Product packaging |       78.31 |             52.29 |            68.14 |

Lower MSE indicates smaller average squared pixel differences from the original image.

For all three test images, reconstruction using the matching blur model produced a lower MSE than both the blurred input and the reconstruction using the smaller, mismatched kernel.

## Observations

- Reconstruction made several prominent edges and text regions appear clearer.
- Small text and fine details were not fully recovered.
- Some edge artifacts were visible in reconstructed images.
- Using a mismatched blur kernel increased MSE relative to the matching-model reconstruction in all three tests.

## Limitations

- The blur was simulated in software, not created by a physical lensless camera.
- The evaluation used only three images, so the findings are preliminary.
- Gaussian blur is a simplified model of real optical blur.
- MSE measures pixel-wise error but does not fully describe perceived image quality.
- Reconstruction did not perfectly recover the original images.

These results demonstrate a small, controlled experiment rather than a general solution to image deblurring.

## How to Run

### 1. Install dependencies

```bash
python -m pip install opencv-python numpy matplotlib
```

### 2. Capture an image

```bash
python capture_image.py
```

Press **S** in the webcam window to save the image as `original.png`.

### 3. Generate a blurred image

```bash
python blur_reconstruction.py
```

### 4. Run reconstruction

```bash
python reconstruct_image.py
```

### 5. Evaluate reconstruction quality

```bash
python evaluate.py
```

### 6. Run the kernel experiment

```bash
python kernel_experiment.py
```

### 7. Evaluate multiple test images

Make sure `test1.png`, `test2.png`, and `test3.png` are available in the project folder.

```bash
python evaluate_dataset.py
python view_results.py
```

Run these commands using a Python environment in which OpenCV, NumPy and Matplotlib are installed.

## Possible Future Work

1. Test additional blur strengths and kernel sizes.
2. Compare reconstruction methods and iteration counts.
3. Evaluate robustness to noise and blur-model mismatch.
4. Explore how computational reconstruction could complement real camera optics or lensless imaging systems.

## Intention

This project is an initial exploration of computational imaging, inverse problems and image reconstruction. It helped me practise building an experiment, measuring results and documenting limitations rather than relying only on visual impressions.
