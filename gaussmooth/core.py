import numpy as np

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    if size <= 0 or sigma <= 0:
        raise ValueError("`size` and `sigma` must be positive.")
    if size % 2 == 0:
        raise ValueError("`size` should be odd to center the kernel symmetrically.")

    half = size // 2
    x = np.linspace(-half, half, size)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

def gaussian_smooth(
    data: np.ndarray,
    sigma: float = 2.0,
    kernel_size: int = 21,
    mode: str = "edge"
) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim != 1:
        raise ValueError("gaussian_smooth only supports 1D arrays.")
    kernel = gaussian_kernel(kernel_size, sigma)
    pad_width = kernel_size // 2
    padded = np.pad(arr, pad_width, mode=mode)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed
  
