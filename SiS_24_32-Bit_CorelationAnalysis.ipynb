import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy, pearsonr
import pydicom
import os
from tqdm import tqdm
import cProfile, pstats

# Constants
PRIME_24BIT = 16777259  # Prime > 2^24
PRIME_32BIT = 4294967291  # Prime > 2^32


def load_image(image_path: str) -> np.ndarray:
    """Load image with automatic type detection."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    ext = os.path.splitext(image_path)[1].lower()
    try:
        if ext == '.dcm':
            ds = pydicom.dcmread(image_path)
            return ds.pixel_array.astype(np.uint16)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Unsupported image format")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        raise RuntimeError(f"Image loading failed: {e}")


def determine_image_type(img: np.ndarray) -> tuple:
    """Return (bit_depth, image_type)"""
    if img.ndim == 3 and img.shape[2] == 3:
        return 24, 'color'
    if img.dtype == np.uint16:
        return 32, 'grayscale'
    return 24, 'grayscale'


def pixels_to_secrets(img: np.ndarray, bit_depth: int) -> list:
    """Convert pixels to integer secrets based on bit depth."""
    if bit_depth == 24 and img.ndim == 3:
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        packed = (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)
        return packed.ravel().tolist()
    if bit_depth == 32:
        flat = img.ravel()
        if len(flat) % 2:
            flat = np.append(flat, 0)
        packed = (flat[::2].astype(np.uint32) << 16) | flat[1::2].astype(np.uint32)
        return packed.tolist()
    return img.ravel().astype(np.uint32).tolist()


def secrets_to_pixels(secrets: list, shape: tuple, bit_depth: int, img_type: str) -> np.ndarray:
    """Reconstruct pixel array from secrets."""
    arr = np.array(secrets, dtype=np.uint32)
    if bit_depth == 24 and img_type == 'color':
        r = ((arr >> 16) & 0xFF).astype(np.uint8)
        g = ((arr >> 8) & 0xFF).astype(np.uint8)
        b = (arr & 0xFF).astype(np.uint8)
        return np.stack((r, g, b), axis=-1).reshape(shape)
    if bit_depth == 32:
        high = (arr >> 16).astype(np.uint16)
        low = (arr & 0xFFFF).astype(np.uint16)
        merged = np.empty(high.size + low.size, dtype=np.uint16)
        merged[::2] = high
        merged[1::2] = low
        return merged[:np.prod(shape)].reshape(shape)
    return arr.astype(np.uint8).reshape(shape)


def generate_shares(secret: int, k: int, n: int, prime: int) -> list:
    """Generate n Shamir shares of a secret with threshold k."""
    coeffs = [secret] + [random.SystemRandom().randint(0, prime-1) for _ in range(k-1)]
    shares = []
    for x in range(1, n+1):
        y = sum(coeff * pow(x, exp, prime) for exp, coeff in enumerate(coeffs)) % prime
        shares.append((x, y))
    return shares


def reconstruct_secret(shares: list, prime: int) -> int:
    """Reconstruct the secret from k shares using Lagrange interpolation."""
    secret = 0
    for i, (xi, yi) in enumerate(shares):
        li = 1
        for j, (xj, _) in enumerate(shares):
            if i != j:
                li = (li * xj * pow(xj - xi, prime-2, prime)) % prime
        secret = (secret + yi * li) % prime
    return secret


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Peak Signal‑to‑Noise Ratio between two images."""
    if img1.dtype != img2.dtype:
        img2 = img2.astype(img1.dtype)
    max_val = 65535 if img1.dtype == np.uint16 else 255
    mse = np.mean((img1 - img2) ** 2)
    return float('inf') if mse == 0 else 20 * np.log10(max_val / np.sqrt(mse))


def evaluate_reconstruction(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """Compute PSNR, SSIM, entropy, histogram and pixel‑intensity correlation."""
    # Align dtype
    if original.dtype != reconstructed.dtype:
        reconstructed = reconstructed.astype(original.dtype)

    # PSNR
    psnr_val = calculate_psnr(original, reconstructed)

    # SSIM & histogram correlation
    if original.ndim == 3 and original.shape[2] == 3:
        ssim_val = ssim(original, reconstructed, channel_axis=2, data_range=255)
        hist_corrs = []
        for c in range(3):
            orig_ch = original[:,:,c].astype(np.uint8)
            rec_ch  = reconstructed[:,:,c].astype(np.uint8)
            h1 = cv2.calcHist([orig_ch], [0], None, [256], [0,256])
            h2 = cv2.calcHist([rec_ch],  [0], None, [256], [0,256])
            hist_corrs.append(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
        hist_corr = float(np.mean(hist_corrs))
    else:
        # Grayscale path
        ssim_val = ssim(original, reconstructed, data_range=255)
        orig_gray = original.astype(np.uint8)
        rec_gray  = reconstructed.astype(np.uint8)
        h1 = cv2.calcHist([orig_gray], [0], None, [256], [0,256])
        h2 = cv2.calcHist([rec_gray],  [0], None, [256], [0,256])
        hist_corr = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))

    # Entropy
    ent_orig = float(entropy(original.ravel(), base=2))
    ent_rec  = float(entropy(reconstructed.ravel(), base=2))

    # Pearson pixel correlation
    orig_flat = original.ravel().astype(np.float64)
    rec_flat  = reconstructed.ravel().astype(np.float64)
    pixel_corr, _ = pearsonr(orig_flat, rec_flat)

    return {
        'psnr': psnr_val,
        'ssim': ssim_val,
        'entropy_orig': ent_orig,
        'entropy_rec': ent_rec,
        'hist_corr': hist_corr,
        'pixel_corr': pixel_corr
    }


def main():
    IMAGE_PATH = '/Users/rmbagdi/Downloads/series-00000/image-00000.dcm'  # update as needed
    K, N = 3, 5

    print("Loading image...")
    img = load_image(IMAGE_PATH)
    bit_depth, img_type = determine_image_type(img)
    prime = PRIME_24BIT if bit_depth == 24 else PRIME_32BIT

    print(f"Processing {img_type} image ({bit_depth}-bit), shape {img.shape}, dtype {img.dtype}")

    print("Converting pixels to secrets...")
    secrets = pixels_to_secrets(img, bit_depth)

    print(f"Generating shares (k={K}, n={N})...")
    share_groups = [generate_shares(s, K, N, prime) for s in tqdm(secrets, desc="Sharing")]

    print("Reconstructing secrets...")
    reconstructed = [reconstruct_secret(sh[:K], prime) for sh in tqdm(share_groups, desc="Reconstructing")]

    print("Rebuilding image...")
    rec_img = secrets_to_pixels(reconstructed, img.shape, bit_depth, img_type)

    print("Evaluating results...")
    metrics = evaluate_reconstruction(img, rec_img)

    # Print metrics
    print("\nQuality Metrics:")
    for k, v in metrics.items():
        print(f"{k.upper():<20}: {v:.4f}" if isinstance(v, float) else f"{k.upper():<20}: {v}")

    # Scatter plot of pixel intensities
    plt.figure(figsize=(6,6))
    plt.scatter(img.ravel(), rec_img.ravel(), s=1, alpha=0.2)
    plt.xlabel('Original Intensity')
    plt.ylabel('Reconstructed Intensity')
    plt.title(f'Pixel Correlation (r={metrics["pixel_corr"]:.4f})')
    plt.tight_layout(); plt.show()

    # Histogram comparison overlay
    plt.figure(figsize=(10,4))
    if img.ndim == 3 and img.shape[2] == 3:
        labels = ['R','G','B']
        for c, col in enumerate(labels):
            o = img[:,:,c].astype(np.uint8); r = rec_img[:,:,c].astype(np.uint8)
            h1 = cv2.calcHist([o], [0], None, [256], [0,256]).flatten()
            h2 = cv2.calcHist([r],  [0], None, [256], [0,256]).flatten()
            plt.plot(h1, linestyle='-')
            plt.plot(h2, linestyle='--')
        plt.legend([f'Orig {l}' for l in labels] + [f'Recon {l}' for l in labels])
    else:
        o = img.astype(np.uint8); r = rec_img.astype(np.uint8)
        h1 = cv2.calcHist([o], [0], None, [256], [0,256]).flatten()
        h2 = cv2.calcHist([r],  [0], None, [256], [0,256]).flatten()
        plt.plot(h1, linestyle='-')
        plt.plot(h2, linestyle='--')
        plt.legend(['Orig','Recon'])
    plt.title('Histogram Comparison')
    plt.xlabel('Bin'); plt.ylabel('Frequency')
    plt.tight_layout(); plt.show()

if __name__ == '__main__':
    cProfile.run('main()', 'profiling_stats')
    stats = pstats.Stats('profiling_stats')
    stats.sort_stats('cumtime').print_stats(10)
