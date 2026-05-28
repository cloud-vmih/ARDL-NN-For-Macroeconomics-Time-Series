from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import VMDConfig


@dataclass
class VariationalModeDecomposer:
    """Small self-contained VMD implementation returning K additive modes."""

    config: VMDConfig = VMDConfig()

    def decompose_array(self, signal: np.ndarray) -> np.ndarray:
        x = np.asarray(signal, dtype=float)
        if x.ndim != 1:
            raise ValueError("VMD expects a one-dimensional signal.")
        if len(x) < 4:
            raise ValueError("VMD requires at least 4 observations.")

        original_len = len(x)
        if original_len % 2:
            x = np.r_[x, x[-1]]

        x = x - np.mean(x)
        mirrored = np.r_[np.flip(x[: len(x) // 2]), x, np.flip(x[len(x) // 2 :])]
        t = len(mirrored)
        freqs = np.arange(t, dtype=float) / t - 0.5 - (1.0 / t)
        f_hat = np.fft.fftshift(np.fft.fft(mirrored))
        f_hat_plus = f_hat.copy()
        f_hat_plus[: t // 2] = 0.0

        k = self.config.modes
        alpha = np.full(k, float(self.config.alpha))
        omega = np.zeros((self.config.max_iterations, k), dtype=float)
        if self.config.init == 1:
            omega[0, :] = 0.5 / k * np.arange(k)
        elif self.config.init == 2:
            rng = np.random.default_rng(42)
            omega[0, :] = np.sort(np.exp(np.log(1.0 / t) + (np.log(0.5) - np.log(1.0 / t)) * rng.random(k)))
        if self.config.dc:
            omega[0, 0] = 0.0

        u_hat_plus = np.zeros((self.config.max_iterations, t, k), dtype=complex)
        lambda_hat = np.zeros((self.config.max_iterations, t), dtype=complex)
        sum_uk = np.zeros(t, dtype=complex)
        u_diff = self.config.tolerance + np.finfo(float).eps
        n = 0

        while u_diff > self.config.tolerance and n < self.config.max_iterations - 1:
            for mode in range(k):
                sum_uk = sum_uk - u_hat_plus[n, :, mode]
                denominator = 1.0 + alpha[mode] * (freqs - omega[n, mode]) ** 2
                u_hat_plus[n + 1, :, mode] = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2.0) / denominator
                sum_uk = sum_uk + u_hat_plus[n + 1, :, mode]

                if not (self.config.dc and mode == 0):
                    spectrum = np.abs(u_hat_plus[n + 1, t // 2 :, mode]) ** 2
                    weight = np.sum(spectrum)
                    if weight > 0:
                        omega[n + 1, mode] = np.sum(freqs[t // 2 :] * spectrum) / weight

            lambda_hat[n + 1, :] = lambda_hat[n, :] + self.config.tau * (sum_uk - f_hat_plus)
            n += 1
            u_diff = np.finfo(float).eps
            for mode in range(k):
                delta = u_hat_plus[n, :, mode] - u_hat_plus[n - 1, :, mode]
                u_diff += (1.0 / t) * np.vdot(delta, delta).real

        u_hat = np.zeros((t, k), dtype=complex)
        u_hat[t // 2 :, :] = u_hat_plus[n, t // 2 :, :]
        u_hat[: t // 2, :] = np.conj(np.flip(u_hat_plus[n, t // 2 :, :], axis=0))
        modes = np.real(np.fft.ifft(np.fft.ifftshift(u_hat, axes=0), axis=0)).T
        modes = modes[:, t // 4 : 3 * t // 4]
        return modes[:, :original_len]

    def decompose(self, series: pd.Series) -> dict[str, pd.Series]:
        values = series.astype(float).to_numpy()

        # decompose_array() đã center chuỗi trước khi tách mode.
        # Vì vậy residue được tính trực tiếp từ chuỗi gốc sẽ tự giữ lại
        # phần mean và phần tín hiệu chưa được các mode giải thích.
        modes = self.decompose_array(values)
        residue = values - modes.sum(axis=0)

        out = {
            f"VMD{index + 1}": pd.Series(
                mode,
                index=series.index,
                name=f"{series.name}_VMD{index + 1}",
            )
            for index, mode in enumerate(modes)
        }

        out["RES"] = pd.Series(
            residue,
            index=series.index,
            name=f"{series.name}_RES",
        )

        return out
    def decompose_frame(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        by_component: dict[str, list[pd.Series]] = {}
        for col in data.columns:
            for component, decomposed in self.decompose(data[col]).items():
                by_component.setdefault(component, []).append(decomposed.rename(col))
        return {name: pd.concat(series_list, axis=1) for name, series_list in by_component.items()}
