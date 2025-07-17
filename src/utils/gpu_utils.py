"""
GPU acceleration utilities for audio and video processing.

This module provides helper functions for GPU-accelerated operations using CuPy
for numerical computations and CUDA-accelerated OpenCV for image processing.
Optimized for WSL environments where OpenCV CUDA support may be limited.
"""

import os
import logging
import numpy as np
import cv2

# Initialize logger
from src.utils.logging_config import setup_logging
logger = setup_logging()

# Try to import PyTorch for CUDA availability check
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        CUDA_DEVICE_NAME = torch.cuda.get_device_name(0)
        logger.info(f"CUDA available: {CUDA_DEVICE_NAME}")
    else:
        logger.warning("CUDA not available through PyTorch")
except ImportError:
    CUDA_AVAILABLE = False
    logger.warning("PyTorch not available, cannot check CUDA status")

# Try to import CuPy, fall back to NumPy if not available
try:
    import cupy as cp
    HAS_CUPY = True
    logger.info("CuPy successfully imported - GPU acceleration available for numerical operations")
    # Set environment variable to control CuPy memory pool
    os.environ["CUPY_GPU_MEMORY_LIMIT"] = "80%" # Use up to 80% of GPU memory for CuPy
except ImportError:
    logger.warning("CuPy not available - falling back to CPU-only NumPy operations")
    cp = np
    HAS_CUPY = False

# Check for CUDA-enabled OpenCV
OPENCV_CUDA_AVAILABLE = cv2.cuda.getCudaEnabledDeviceCount() > 0
if OPENCV_CUDA_AVAILABLE:
    logger.info(f"CUDA-enabled OpenCV detected with {cv2.cuda.getCudaEnabledDeviceCount()} device(s)")
else:
    logger.warning("CUDA-enabled OpenCV not available - falling back to CPU operations")
    # In WSL environments, this is common even when CUDA is available through other libraries

def get_array_module(x):
    """
    Return the appropriate array module (NumPy or CuPy) for the input array.
    
    Args:
        x: Input array (NumPy or CuPy)
        
    Returns:
        Module to use for array operations
    """
    if HAS_CUPY and isinstance(x, cp.ndarray):
        return cp
    return np

def to_gpu(array):
    """
    Transfer a NumPy array to GPU if CuPy is available.
    
    Args:
        array: NumPy array
        
    Returns:
        CuPy array if GPU is available, otherwise the original NumPy array
    """
    if HAS_CUPY:
        try:
            return cp.asarray(array)
        except Exception as e:
            logger.warning(f"Failed to transfer array to GPU: {e}")
    return array

def to_cpu(array):
    """
    Transfer a CuPy array to CPU.
    
    Args:
        array: CuPy or NumPy array
        
    Returns:
        NumPy array
    """
    if HAS_CUPY and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return array

def process_image_gpu(image):
    """
    Transfer an image to GPU for processing with CUDA-accelerated OpenCV or CuPy.
    
    Args:
        image: NumPy array representing an image
        
    Returns:
        GpuMat object if CUDA OpenCV is available, CuPy array if only CuPy is available,
        otherwise the original image
    """
    # First try OpenCV CUDA if available
    if OPENCV_CUDA_AVAILABLE:
        try:
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(image)
            return gpu_mat
        except Exception as e:
            logger.warning(f"Failed to upload image to GPU via OpenCV: {e}")
    
    # If OpenCV CUDA failed but CuPy is available, use CuPy
    if HAS_CUPY and not OPENCV_CUDA_AVAILABLE:
        try:
            return cp.asarray(image)
        except Exception as e:
            logger.warning(f"Failed to upload image to GPU via CuPy: {e}")
    
    return image

def download_image_gpu(gpu_mat):
    """
    Download a processed image from GPU.
    
    Args:
        gpu_mat: GpuMat object, CuPy array, or NumPy array
        
    Returns:
        NumPy array representing the image
    """
    # Handle OpenCV CUDA GpuMat
    if OPENCV_CUDA_AVAILABLE and isinstance(gpu_mat, cv2.cuda_GpuMat):
        try:
            return gpu_mat.download()
        except Exception as e:
            logger.warning(f"Failed to download image from GPU via OpenCV: {e}")
            return np.array(gpu_mat)
    
    # Handle CuPy array
    if HAS_CUPY and isinstance(gpu_mat, cp.ndarray):
        try:
            return cp.asnumpy(gpu_mat)
        except Exception as e:
            logger.warning(f"Failed to download image from GPU via CuPy: {e}")
    
    # Already a NumPy array or other type
    return gpu_mat

def apply_filter_gpu(image, filter_func, *args, **kwargs):
    """
    Apply a filter to an image using GPU acceleration if available.
    Optimized for WSL environments where OpenCV CUDA may not be available but CuPy is.
    
    Args:
        image: Input image (NumPy array)
        filter_func: Filter function to apply
        *args, **kwargs: Arguments to pass to the filter function
        use_cupy: Whether to try using CuPy if OpenCV CUDA fails (default: True)
        
    Returns:
        Filtered image (NumPy array)
    """
    # Extract use_cupy parameter if provided, default to True
    use_cupy = kwargs.pop('use_cupy', True)
    
    # Try OpenCV CUDA first if available
    if OPENCV_CUDA_AVAILABLE:
        try:
            # Upload to GPU
            gpu_mat = process_image_gpu(image)
            
            # Apply filter
            result = filter_func(gpu_mat, *args, **kwargs)
            
            # Download result
            return download_image_gpu(result)
        except Exception as e:
            logger.warning(f"OpenCV CUDA filter application failed: {e}. Trying CuPy or falling back to CPU.")
    
    # If OpenCV CUDA failed or not available, try CuPy if requested
    if HAS_CUPY and use_cupy:
        try:
            # Convert to CuPy array
            gpu_array = to_gpu(image)
            
            # Apply filter with CuPy array
            result = filter_func(gpu_array, *args, **kwargs)
            
            # Convert back to NumPy
            return to_cpu(result)
        except Exception as e:
            logger.warning(f"CuPy filter application failed: {e}. Falling back to CPU.")
    
    # Fall back to CPU
    return filter_func(image, *args, **kwargs)

def apply_canny_edge_detection(frame, low_threshold=100, high_threshold=200, use_gpu=True):
    """
    Apply Canny edge detection to a frame with GPU acceleration if available.
    Optimized for WSL environments where OpenCV CUDA may not be available but CuPy is.
    
    Args:
        frame: Input image frame (NumPy array)
        low_threshold: Lower threshold for the hysteresis procedure
        high_threshold: Higher threshold for the hysteresis procedure
        use_gpu: Whether to use GPU acceleration if available
        
    Returns:
        Edge detection result as RGB image (NumPy array)
    """
    # If GPU acceleration is disabled, use CPU directly
    if not use_gpu:
        # Convert frame to grayscale if it's RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return edges_rgb
    
    # Try OpenCV CUDA first if available
    if OPENCV_CUDA_AVAILABLE:
        try:
            # Convert frame to grayscale if it's RGB
            gpu_frame = process_image_gpu(frame)
            
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
            else:
                gpu_gray = gpu_frame
                
            gpu_edges = cv2.cuda.Canny(gpu_gray, low_threshold, high_threshold)
            edges = download_image_gpu(gpu_edges)
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return edges_rgb
        except Exception as e:
            logger.warning(f"OpenCV CUDA Canny edge detection failed: {e}. Trying CuPy or falling back to CPU.")
    
    # If OpenCV CUDA failed or not available, try CuPy if available
    if HAS_CUPY:
        try:
            # Convert to CuPy array
            gpu_frame = to_gpu(frame)
            
            # Convert frame to grayscale if it's RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Use NumPy for color conversion since CuPy doesn't have direct cvtColor
                gray = cv2.cvtColor(to_cpu(gpu_frame), cv2.COLOR_BGR2GRAY)
                gpu_gray = to_gpu(gray)
            else:
                gpu_gray = gpu_frame
            
            # Download for OpenCV Canny (no direct CuPy implementation)
            gray = to_cpu(gpu_gray)
            edges = cv2.Canny(gray, low_threshold, high_threshold)
            
            # Upload result back to GPU for any further processing
            gpu_edges = to_gpu(edges)
            
            # Download and convert to RGB
            edges = to_cpu(gpu_edges)
            edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return edges_rgb
        except Exception as e:
            logger.warning(f"CuPy Canny edge detection failed: {e}. Falling back to CPU.")
    
    # Fall back to CPU
    # Convert frame to grayscale if it's RGB
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
        
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_rgb
