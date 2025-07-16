"""
GPU acceleration utilities for audio and video processing.

This module provides helper functions for GPU-accelerated operations using CuPy
for numerical computations and CUDA-accelerated OpenCV for image processing.
"""

import os
import logging
import numpy as np
import cv2

# Initialize logger
from src.utils.logging_config import setup_logging
logger = setup_logging()

# Try to import CuPy, fall back to NumPy if not available
try:
    import cupy as cp
    HAS_CUPY = True
    logger.info("CuPy successfully imported - GPU acceleration available for numerical operations")
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

# Set environment variable to control CuPy memory pool
os.environ["CUPY_GPU_MEMORY_LIMIT"] = "90%" # Use up to 90% of GPU memory

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
    Transfer an image to GPU for processing with CUDA-accelerated OpenCV.
    
    Args:
        image: NumPy array representing an image
        
    Returns:
        GpuMat object if CUDA OpenCV is available, otherwise the original image
    """
    if OPENCV_CUDA_AVAILABLE:
        gpu_mat = cv2.cuda_GpuMat()
        gpu_mat.upload(image)
        return gpu_mat
    return image

def download_image_gpu(gpu_mat):
    """
    Download a processed image from GPU.
    
    Args:
        gpu_mat: GpuMat object or NumPy array
        
    Returns:
        NumPy array representing the image
    """
    if OPENCV_CUDA_AVAILABLE and isinstance(gpu_mat, cv2.cuda_GpuMat):
        return gpu_mat.download()
    return gpu_mat

def apply_filter_gpu(image, filter_func, *args, **kwargs):
    """
    Apply a filter to an image using GPU acceleration if available.
    
    Args:
        image: Input image (NumPy array)
        filter_func: Filter function to apply
        *args, **kwargs: Arguments to pass to the filter function
        
    Returns:
        Filtered image (NumPy array)
    """
    if OPENCV_CUDA_AVAILABLE:
        gpu_image = process_image_gpu(image)
        result = filter_func(gpu_image, *args, **kwargs)
        return download_image_gpu(result)
    else:
        return filter_func(image, *args, **kwargs)
