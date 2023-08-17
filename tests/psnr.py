import cv2
import fire
import numpy as np


def calculate_psnr(img1_path, img2_path):
    # Read the images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Check if the images were loaded correctly
    if img1 is None:
        raise ValueError(f"Image at {img1_path} could not be loaded.")
    if img2 is None:
        raise ValueError(f"Image at {img2_path} could not be loaded.")

    # Ensure the dimensions of the two images match
    if img1 is not None and img2 is not None and img1.shape != img2.shape:
        raise ValueError("The dimensions of the two images do not match.")

    # Calculate Mean Squared Error (MSE)
    mse = np.mean((img1 - img2) ** 2)

    # If MSE is 0, PSNR is infinite (images are identical)
    if mse == 0:
        return float("inf")

    # Calculate PSNR
    max_pixel = 255.0  # Maximum pixel value (for 8-bit images)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def main(
    img1: str,
    img2: str,
):
    try:
        psnr_value = calculate_psnr(img1, img2)
        if psnr_value < 30:
            raise ValueError(f"PSNR between {img1} and {img2} is too low: {psnr_value:.2f} dB")
        print(f"PSNR between {img1} and {img2}: {psnr_value:.2f} dB")
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    fire.Fire(main)
