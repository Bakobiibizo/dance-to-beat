"""
Circle Image Creator

This script provides functionality to create circular masks for images with smooth edges
and proper padding. It also ensures images are square by adding padding when necessary.

Key Features:
    - Create circular masks with adjustable padding
    - Convert rectangular images to square with centered content
    - Smooth edge handling for circular masks
    - Support for various image formats (PNG, JPEG, WEBP)

Example:
    python circle_image.py --image "./media/image.jpg" --padding 32
"""

import cv2
import numpy as np
import os

def create_circular_mask(height, width, padding=16):
    """Create a circular mask with smooth edges and specified padding.
    
    Args:
        height (int): Height of the mask in pixels
        width (int): Width of the mask in pixels
        padding (int, optional): Padding from edges in pixels. Defaults to 16.
    
    Returns:
        numpy.ndarray: 8-bit single channel mask with smooth edges
    """
    # Create meshgrid of coordinates
    Y, X = np.mgrid[:height, :width]
    
    # Calculate center
    center = (width // 2, height // 2)
    
    # Calculate radius (use the smaller dimension and subtract padding)
    radius = min(center[0], center[1]) - padding
    
    # Calculate distance from center for each pixel
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    # Create mask with smooth edges
    mask = np.zeros((height, width), dtype=np.float32)
    
    # Create smooth falloff near the edge
    edge_width = 2
    inner_radius = radius - edge_width
    
    # Set inner circle to fully opaque
    mask[dist_from_center <= inner_radius] = 255
    
    # Create smooth transition at the edge
    edge_region = (dist_from_center > inner_radius) & (dist_from_center <= radius)
    mask[edge_region] = 255 * (1 - (dist_from_center[edge_region] - inner_radius) / edge_width)
    
    return np.clip(mask, 0, 255).astype(np.uint8)


def make_square_image(image):
    """Make the image square by adding black padding to the shorter dimension.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
    
    Returns:
        numpy.ndarray: Square image with the original content centered
    """
    height, width = image.shape[:2]
    size = max(height, width)
    
    # Create a black square image
    square = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Calculate padding
    y_offset = (size - height) // 2
    x_offset = (size - width) // 2
    
    # Place the original image in the center
    square[y_offset:y_offset+height, x_offset:x_offset+width] = image
    
    return square


def main(image_path="./media/spiral.jpeg", padding=16):
    """Create a circular mask for the input image and save the result.
    
    Args:
        image_path (str, optional): Path to input image. Defaults to "./media/spiral.jpeg".
        padding (int, optional): Padding from edges in pixels. Defaults to 16.
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return

    # Load the image first
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return
    
    # Make the image square
    image = make_square_image(image)
    height, width = image.shape[:2]
    
    # Create mask with specified padding
    print(f"Creating circular mask with {padding}px padding")
    mask = create_circular_mask(height, width, padding=padding)
    
    # Apply the mask
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Create output directory if it doesn't exist
    output_dir = "./media"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the result
    output_path = os.path.join(output_dir, "masked_image.png")
    cv2.imwrite(output_path, masked_image)
    print(f"Saved masked image to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create a circular mask for an image')
    parser.add_argument('--image', default='./media/spiral.jpeg', help='Path to the input image')
    parser.add_argument('--padding', type=int, default=16, help='Padding from edges in pixels')
    args = parser.parse_args()
    main(args.image, args.padding)