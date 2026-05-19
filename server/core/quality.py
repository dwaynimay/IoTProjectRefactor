# File: server/core/quality.py

# =============================================================================
# quality.py — Metrik Kualitas Rekonstruksi CS
# =============================================================================
#
# Dihitung SETELAH rekonstruksi selesai (x_hat sudah tersedia).
# Tidak membutuhkan sinyal asli — semua metrik berbasis measurement residual.
#
# CARA PAKAI:
#   from core.quality import ReconQuality, assess
#
#   report = assess(y=y_vector, x_hat=x_hat, phi=PHI)
#   if report.is_low_quality:
#       print(f"WARNING: {report.summary}")
#
# METRIK:
#   residual_norm     : ||y - Φx̂||₂  (semakin kecil semakin baik)
#   measurement_norm  : ||y||₂
#   relative_error    : residual_norm / measurement_norm  (0.0 = sempurna)
#   sparsity_ratio    : fraksi koefisien DCT ≠ 0  (harusnya kecil, < 0.5)
#   is_low_quality    : True jika relative_error > threshold
#
# THRESHOLD default di config.py dapat di-override via parameter.
# =============================================================================

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# Threshold default — bisa di-override di config.py
DEFAULT_RELATIVE_ERROR_THRESHOLD = 0.25   # > 25% residual = LOW_QUALITY
DEFAULT_SPARSITY_WARN_THRESHOLD  = 0.60   # > 60% koefisien ≠ 0 = suspiciously dense

# Nilai minimum ||y|| untuk menghindari divisi by zero
_MIN_NORM = 1e-8


# ---------------------------------------------------------------------------
# Data class hasil assessment
# ---------------------------------------------------------------------------

@dataclass
class ReconQuality:
    signal_name       : str
    residual_norm     : float
    measurement_norm  : float
    relative_error    : float          # residual / measurement norm
    sparsity_ratio    : float          # fraksi koef ≠ 0
    n_nonzero         : int
    n_total           : int
    is_low_quality    : bool
    warnings          : list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        tag = "LOW_QUALITY" if self.is_low_quality else "OK"
        return (
            f"[{self.signal_name}] {tag} | "
            f"rel_err={self.relative_error:.3f} | "
            f"sparsity={self.sparsity_ratio:.2f} ({self.n_nonzero}/{self.n_total}) | "
            f"resid={self.residual_norm:.4f} / meas={self.measurement_norm:.4f}"
        )

    @property
    def as_dict(self) -> dict:
        return {
            "signal"          : self.signal_name,
            "relative_error"  : round(self.relative_error, 4),
            "sparsity_ratio"  : round(self.sparsity_ratio, 4),
            "n_nonzero"       : self.n_nonzero,
            "n_total"         : self.n_total,
            "residual_norm"   : round(self.residual_norm, 6),
            "measurement_norm": round(self.measurement_norm, 6),
            "is_low_quality"  : self.is_low_quality,
            "warnings"        : self.warnings,
        }


# ---------------------------------------------------------------------------
# Core assessment function
# ---------------------------------------------------------------------------

def assess(
    y          : "list | np.ndarray",
    x_hat      : np.ndarray,
    phi        : np.ndarray,
    signal_name: str = "signal",
    err_thresh : float = DEFAULT_RELATIVE_ERROR_THRESHOLD,
    sparsity_thresh: float = DEFAULT_SPARSITY_WARN_THRESHOLD,
    zero_eps   : float = 1e-6,
) -> ReconQuality:
    """
    Hitung metrik kualitas rekonstruksi satu sinyal.

    Args:
        y           : measurement vector (m,) — dari sensor
        x_hat       : sinyal rekonstruksi (n,) — output reconstruct()
        phi         : matriks pengukuran Φ (m × n)
        signal_name : nama sinyal untuk logging
        err_thresh  : batas relative_error untuk flag LOW_QUALITY
        sparsity_thresh: batas sparsity_ratio untuk warning
        zero_eps    : threshold anggap koefisien = 0

    Returns:
        ReconQuality dataclass
    """
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(x_hat, dtype=np.float64)

    # Residual: y - Φx̂
    y_hat = phi @ x_arr
    residual = y_arr - y_hat

    residual_norm    = float(np.linalg.norm(residual))
    measurement_norm = float(np.linalg.norm(y_arr))

    # Hindari div-by-zero
    if measurement_norm < _MIN_NORM:
        relative_error = 0.0 if residual_norm < _MIN_NORM else float("inf")
    else:
        relative_error = residual_norm / measurement_norm

    # Sparsity (dalam domain DCT — x_hat adalah koefisien sparse)
    n_total   = len(x_arr)
    n_nonzero = int(np.sum(np.abs(x_arr) > zero_eps))
    sparsity_ratio = n_nonzero / n_total if n_total > 0 else 0.0

    # Kumpulkan warnings
    warnings: list[str] = []

    if not math.isfinite(relative_error):
        warnings.append("measurement_norm mendekati 0 — sinyal mungkin nol")

    if sparsity_ratio > sparsity_thresh:
        warnings.append(
            f"sparsity_ratio={sparsity_ratio:.2f} > {sparsity_thresh} "
            f"— sinyal tidak sparse, kualitas OMP mungkin rendah"
        )

    if relative_error > 0.5:
        warnings.append(
            f"relative_error={relative_error:.3f} sangat tinggi — "
            "periksa sinkronisasi matriks Φ antara firmware dan server"
        )

    is_low_quality = relative_error > err_thresh

    return ReconQuality(
        signal_name      = signal_name,
        residual_norm    = residual_norm,
        measurement_norm = measurement_norm,
        relative_error   = relative_error,
        sparsity_ratio   = sparsity_ratio,
        n_nonzero        = n_nonzero,
        n_total          = n_total,
        is_low_quality   = is_low_quality,
        warnings         = warnings,
    )


# ---------------------------------------------------------------------------
# Batch assessment untuk window penuh (semua sinyal sekaligus)
# ---------------------------------------------------------------------------

def assess_window(
    measurements : dict[str, "list | np.ndarray"],
    reconstructed: dict[str, np.ndarray],
    phi          : np.ndarray,
    err_thresh   : float = DEFAULT_RELATIVE_ERROR_THRESHOLD,
    sparsity_thresh: float = DEFAULT_SPARSITY_WARN_THRESHOLD,
) -> dict[str, ReconQuality]:
    """
    Assess kualitas semua sinyal dalam satu window sekaligus.

    Args:
        measurements  : {signal_name: y_vector}  — measurement vectors
        reconstructed : {signal_name: x_hat}     — hasil reconstruct()
        phi           : matriks Φ yang digunakan

    Returns:
        dict {signal_name: ReconQuality}
    """
    results: dict[str, ReconQuality] = {}
    for sig, x_hat in reconstructed.items():
        y = measurements.get(sig)
        if y is None:
            continue
        results[sig] = assess(
            y=y,
            x_hat=x_hat,
            phi=phi,
            signal_name=sig,
            err_thresh=err_thresh,
            sparsity_thresh=sparsity_thresh,
        )
    return results


def window_summary(quality_map: dict[str, ReconQuality]) -> dict:
    """
    Ringkasan window: avg relative_error, flag any LOW_QUALITY, total warnings.
    """
    if not quality_map:
        return {"ok": True, "signals": {}, "avg_relative_error": 0.0}

    errors = [q.relative_error for q in quality_map.values()
              if math.isfinite(q.relative_error)]
    avg_err = sum(errors) / len(errors) if errors else 0.0
    any_low = any(q.is_low_quality for q in quality_map.values())
    all_warnings = [w for q in quality_map.values() for w in q.warnings]

    return {
        "ok"                 : not any_low,
        "avg_relative_error" : round(avg_err, 4),
        "any_low_quality"    : any_low,
        "low_quality_signals": [s for s, q in quality_map.items() if q.is_low_quality],
        "warnings"           : all_warnings,
        "signals"            : {s: q.as_dict for s, q in quality_map.items()},
    }
