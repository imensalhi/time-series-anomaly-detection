from __future__ import annotations

import numpy as np
import pywt


class CWTService:
    def __init__(self, scales_min: int = 1, scales_max: int = 65, wavelet: str = "morl") -> None:
        self.scales = np.arange(scales_min, scales_max)
        self.wavelet = wavelet

    def transform(self, windows: np.ndarray) -> np.ndarray:
        specs = []
        for window in windows:
            coeffs, _ = pywt.cwt(window, self.scales, self.wavelet)
            specs.append(np.abs(coeffs))
        arr = np.array(specs, dtype=np.float32)
        return arr
