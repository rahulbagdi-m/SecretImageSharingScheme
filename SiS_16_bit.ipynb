import numpy as np 
import cv2
import random
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
import pydicom
import os


def load_image(image_path, raw_dims=None):
    ext = os.path.splitext(image_path)[1].lower()
    
    if ext == '.dcm':
        dicom_data = pydicom.dcmread(image_path)
        # For DICOM, assume pixel_array is already in desired bit depth.
        image = dicom_data.pixel_array.astype(np.uint16)
    elif ext == '.raw':
        if raw_dims is None:
            raise ValueError("For raw images, please provide raw_dims as a tuple (height, width).")
        with open(image_path, 'rb') as f:
            raw = np.frombuffer(f.read(), dtype=np.uint16)
        image = raw.reshape(raw_dims)
    else:
        # Use cv2.IMREAD_UNCHANGED to preserve the original bit depth.
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise FileNotFoundError("Image not found or could not be loaded. Check file path/format.")
    
    # For consistency, if image is not 16-bit, convert it appropriately.
    if image.dtype != np.uint16:
        image = cv2.normalize(image, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)
    
    return image

def modinv(a, mod):
    """
    Compute modular inverse using the Extended Euclidean Algorithm.
    """
    a = a % mod
    if a == 0:
        raise ValueError("Inverse does not exist")
    lm, hm = 1, 0
    low, high = a, mod
    while low > 1:
        ratio = high // low
        nm = hm - lm * ratio
        new = high - low * ratio
        lm, low, hm, high = nm, new, lm, low
    return lm % mod

# ---------------------------
# Secret Sharing Functions for GF(4294967311) (32-bit version)
# ---------------------------
def share_secret_32(secret, k, n, prime=4294967311):
    """
    Share a single 32-bit secret using a (k, n)-threshold scheme.
    Returns a list of (x, y) shares.
    """
    coeffs = [secret] + [random.randint(0, prime - 1) for _ in range(k - 1)]
    shares = []
    for i in range(1, n + 1):
        x = i
        y = sum(coeff * pow(x, j, prime) % prime for j, coeff in enumerate(coeffs)) % prime
        if y == prime:
            y = 0
        shares.append((x, y))
    return shares

def reconstruct_secret_32(shares, prime=4294967311):
    """
    Reconstruct the 32-bit secret using Lagrange interpolation from a list of (x, y) shares.
    """
    secret = 0
    k = len(shares)
    for i, (xi, yi) in enumerate(shares):
        li = 1
        for j, (xj, _) in enumerate(shares):
            if i != j:
                li = (li * xj * modinv(xj - xi, prime)) % prime
        secret = (secret + yi * li) % prime
    return secret

# ---------------------------
# Pixel Linking/Delinking for 32-bit Encoding
# ---------------------------
def pixel_link_32(image):
    """
    Link adjacent 16-bit pixels to form 32-bit secret values.
    Assumes that the input image is in 16-bit format (np.uint16).
    """
    flat = image.flatten()
    if len(flat) % 2 != 0:
        flat = np.append(flat, 0)
    # Each secret = 65536 * pixel1 + pixel2
    return [65536 * int(flat[i]) + int(flat[i+1]) for i in range(0, len(flat), 2)]

def pixel_delink_32(secret_values):
    """
    Delink 32-bit secret values back into two 16-bit pixel values each.
    """
    pixels = []
    for val in secret_values:
        high_value = val // 65536
        low_value = val % 65536
        pixels.append(high_value)
        pixels.append(low_value)
    return np.array(pixels, dtype=np.uint16)

# ---------------------------
# Evaluation Functions
# ---------------------------
def calculate_entropy(image):
    """
    Compute entropy of a grayscale image.
    """
    hist = np.histogram(image, bins=65536, range=(0, 65535))[0]
    hist = hist / hist.sum()
    return entropy(hist, base=2)

def histogram_correlation(img1, img2):
    hist1 = cv2.calcHist([img1], [0], None, [65536], [0, 65535])
    hist2 = cv2.calcHist([img2], [0], None, [65536], [0, 65535])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def evaluate_image_quality_32(original, reconstructed):
    """
    Compute PSNR, SSIM, histogram correlation, and entropies for 16-bit images.
    Returns the computed metrics.
    """
    # For 16-bit images, adjust data_range accordingly
    psnr_value = cv2.PSNR(original, reconstructed)
    ssim_value = ssim(original, reconstructed, data_range=original.max() - original.min())
    entropy_original = calculate_entropy(original)
    entropy_reconstructed = calculate_entropy(reconstructed)
    hist_corr = histogram_correlation(original, reconstructed)
    return psnr_value, ssim_value, entropy_original, entropy_reconstructed, hist_corr

# ---------------------------
# Main Function for 32-bit Encoding
# ---------------------------
def main():
    # Update this path to your 16-bit image file (e.g., a 16-bit TIFF or DICOM)
    image_path = '/Users/pathToImage/sample_image.jpg'
    image = load_image(image_path)
    
    # Use 32-bit encoding: link adjacent 16-bit pixels
    secrets = pixel_link_32(image)
    
    # Set threshold parameters (k and n)
    k, n = 3, 5
    
    # Generate shares for each secret using the 32-bit scheme
    all_shares = [share_secret_32(secret, k, n) for secret in secrets]
    
    # Reconstruct each secret using the first k shares (or a random selection of k shares)
    recovered_secrets = [reconstruct_secret_32(shares[:k]) for shares in all_shares]
    
    # Convert the recovered 32-bit secrets back to two 16-bit pixels
    recovered_pixels = pixel_delink_32(recovered_secrets)
    recovered_pixels = recovered_pixels[:image.size]  # Ensure correct length
    recovered_image = recovered_pixels.reshape(image.shape)
    
    # Display original and reconstructed images side by side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original 16-bit Image")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(recovered_image, cmap='gray')
    plt.title("Reconstructed Image (32-bit Encoding)")
    plt.axis('off')
    plt.show()
    
    # Evaluate quality metrics for the 16-bit images
    metrics = evaluate_image_quality_32(image, recovered_image)
    print(f"PSNR: {metrics[0]:.4f}")
    print(f"SSIM: {metrics[1]:.4f}")
    print(f"Entropy (Original): {metrics[2]:.4f}")
    print(f"Entropy (Reconstructed): {metrics[3]:.4f}")
    print(f"Histogram Correlation: {metrics[4]:.4f}")

if __name__ == '__main__':
    main()
