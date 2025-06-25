# 32-bit Secret Sharing and Image Reconstruction

This project implements a method for sharing and reconstructing secrets using 32-bit encoding with 16-bit images. The process involves encoding pixel pairs from a 16-bit image into 32-bit values, performing secret sharing using a (k, n)-threshold scheme, and then reconstructing the original image from the shared secrets.

## Features

- **Image Loading**: Supports loading images in various formats including DICOM, RAW, PNG, JPEG, and BMP.
- **Secret Sharing**: Implements a (k, n)-threshold secret sharing scheme for 32-bit values.
- **Pixel Linking/Delinking**: Links adjacent 16-bit pixels to form 32-bit secret values and can delink them back to 16-bit pixels.
- **Image Evaluation**: Calculates various metrics such as PSNR, SSIM, entropy, and histogram correlation to evaluate the quality of the reconstructed image.
- **Visualization**: Displays the original and reconstructed images side-by-side for comparison.

## Requirements

- Python 3.x
- NumPy
- OpenCV (`cv2`)
- Matplotlib
- PyDICOM
- scikit-image
- SciPy

You can install the required dependencies using `pip`:

```bash
pip install numpy opencv-python matplotlib pydicom scikit-image scipy
