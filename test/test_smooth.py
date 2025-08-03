import numpy as np
import pytest
from gaussmooth import gaussian_kernel, gaussian_smooth

def test_kernel_is_normalized():
    kern = gaussian_kernel(size=11, sigma=2.0)
    assert np.isclose(kern.sum(), 1.0), "Kernel should sum to 1"

def test_kernel_symmetry():
    kern = gaussian_kernel(size=11, sigma=2.0)
    # symmetric around center
    assert np.allclose(kern, kern[::-1]), "Kernel should be symmetric"

def test_smooth_preserves_length():
    x = np.random.randn(100)
    y = gaussian_smooth(x, sigma=1.5, kernel_size=11)
    assert len(y) == len(x), "Smoothed output must have same length as input"

def test_constant_signal_stays_constant():
    x = np.ones(50)
    y = gaussian_smooth(x, sigma=3, kernel_size=15)
    assert np.allclose(y, 1.0, atol=1e-8), "Constant signal should remain (approximately) constant"

def test_small_sigma_near_identity():
    x = np.random.randn(100)
    # very small sigma and kernel size 3 should be close to original
    y = gaussian_smooth(x, sigma=0.1, kernel_size=3)
    assert np.allclose(x, y, atol=1e-2), "With tiny smoothing, output should be near input"

def test_invalid_kernel_size_even():
    with pytest.raises(ValueError):
        gaussian_kernel(size=10, sigma=1.0)  # even size not allowed

def test_invalid_parameters():
    with pytest.raises(ValueError):
        gaussian_kernel(size=5, sigma=0)  # sigma must be positive
    with pytest.raises(ValueError):
        gaussian_smooth(np.zeros(10), sigma=1.0, kernel_size=0)  # size must be positive/odd
