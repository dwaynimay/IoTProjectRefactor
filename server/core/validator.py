# File: server/core/validator.py

# =============================================================================
# validator.py — Payload Validator untuk MQTT CS Data
# =============================================================================
#
# Validasi setiap MQTT payload SEBELUM direkonstruksi.
# Semua validasi bersifat strict: gagal 1 field = payload ditolak.
#
# CARA PAKAI di apps/:
#   from core.validator import validate_imu, validate_ppg, ValidationError
#
#   ok, errors = validate_imu(payload, node_id=1)
#   if not ok:
#       log_and_skip(errors)
#       return
#
# SCHEMA YANG DIVALIDASI:
#   IMU payload  : ts, ax, ay, az, gx, gy, gz  (masing-masing list[float] len=CS_M)
#   PPG payload  : ts, ir (list[float] len=CS_M), hr (int), finger (bool)
#
# VALIDASI:
#   1. Schema      : field wajib ada, tipe benar
#   2. Length      : len(signal) == CS_M
#   3. Finite      : tidak ada NaN / Inf dalam measurement vector
#   4. Monotonicity: ts tidak boleh < ts sebelumnya (per-node state)
#   5. Node whitelist: node_id harus terdaftar (opsional)
# =============================================================================

from __future__ import annotations

import math
import time
from typing import Optional

from .config import CS_M, IMU_SIGNALS, PPG_SIGNALS

# ---------------------------------------------------------------------------
# State global: last timestamp per node, per signal type
# ---------------------------------------------------------------------------
_last_ts: dict[tuple[int, str], int] = {}  # key: (node_id, sig_type)

# ---------------------------------------------------------------------------
# Konfigurasi whitelist node (kosong = terima semua node)
# ---------------------------------------------------------------------------
ALLOWED_NODE_IDS: set[int] = set()  # set() = allow all


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _check_finite(values: list, field: str) -> list[str]:
    """Return list error jika ada NaN/Inf dalam vector."""
    errors = []
    for i, v in enumerate(values):
        if not math.isfinite(v):
            errors.append(f"{field}[{i}]={v} tidak finite (NaN/Inf)")
            break  # cukup lapor 1x per sinyal
    return errors


def _check_length(values: list, field: str, expected: int) -> list[str]:
    if len(values) != expected:
        return [f"{field}: panjang {len(values)}, diharapkan {expected}"]
    return []


def _check_ts_monotonic(node_id: int, sig_type: str, ts: int) -> list[str]:
    """Cek ts tidak mundur. Update state jika valid."""
    key = (node_id, sig_type)
    last = _last_ts.get(key, -1)
    if ts < last:
        return [
            f"ts={ts} mundur dari ts sebelumnya {last} "
            f"(diff={last - ts}ms) — kemungkinan replay/reboot"
        ]
    _last_ts[key] = ts
    return []


def _check_node_whitelist(node_id: int) -> list[str]:
    if ALLOWED_NODE_IDS and node_id not in ALLOWED_NODE_IDS:
        return [f"Node ID {node_id} tidak ada dalam whitelist {ALLOWED_NODE_IDS}"]
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_imu(payload: dict, node_id: int = -1) -> tuple[bool, list[str]]:
    """
    Validasi payload cs_imu.

    Args:
        payload  : dict hasil json.loads() dari MQTT
        node_id  : ID node pengirim (untuk monotonicity check)

    Returns:
        (True, [])           jika valid
        (False, [str, ...])  jika invalid, errors berisi deskripsi masalah
    """
    errors: list[str] = []

    # 0. Node whitelist
    if node_id >= 0:
        errors += _check_node_whitelist(node_id)

    # 1. Schema: field wajib
    required_fields = {"ts"} | set(IMU_SIGNALS)
    missing = required_fields - payload.keys()
    if missing:
        errors.append(f"Field wajib tidak ada: {sorted(missing)}")
        return False, errors  # tidak bisa lanjut tanpa field

    # 2. Timestamp: harus int/float positif
    ts = payload.get("ts", 0)
    if not isinstance(ts, (int, float)) or ts < 0:
        errors.append(f"ts={ts!r} bukan angka positif")

    # 3. Validasi setiap sinyal IMU
    for sig in IMU_SIGNALS:
        values = payload.get(sig, [])
        if not isinstance(values, list):
            errors.append(f"{sig} bukan list, dapat {type(values).__name__}")
            continue
        errors += _check_length(values, sig, CS_M)
        if not errors:  # cek finite hanya jika panjang benar
            errors += _check_finite(values, sig)

    # 4. Timestamp monotonicity (per node)
    if node_id >= 0 and isinstance(ts, (int, float)) and ts >= 0:
        errors += _check_ts_monotonic(node_id, "imu", int(ts))

    ok = len(errors) == 0
    return ok, errors


def validate_ppg(payload: dict, node_id: int = -1) -> tuple[bool, list[str]]:
    """
    Validasi payload cs_ppg.

    Args:
        payload  : dict hasil json.loads() dari MQTT
        node_id  : ID node pengirim

    Returns:
        (True, [])           jika valid
        (False, [str, ...])  jika invalid
    """
    errors: list[str] = []

    # 0. Node whitelist
    if node_id >= 0:
        errors += _check_node_whitelist(node_id)

    # 1. Schema: field wajib
    required_fields = {"ts", "ir", "hr", "finger"}
    missing = required_fields - payload.keys()
    if missing:
        errors.append(f"Field wajib tidak ada: {sorted(missing)}")
        return False, errors

    # 2. Timestamp
    ts = payload.get("ts", 0)
    if not isinstance(ts, (int, float)) or ts < 0:
        errors.append(f"ts={ts!r} bukan angka positif")

    # 3. IR measurement vector
    ir = payload.get("ir", [])
    if not isinstance(ir, list):
        errors.append(f"ir bukan list, dapat {type(ir).__name__}")
    else:
        errors += _check_length(ir, "ir", CS_M)
        if not errors:
            errors += _check_finite(ir, "ir")

    # 4. HR: int, range 0-300 (0 = tidak terdeteksi)
    hr = payload.get("hr", -1)
    if not isinstance(hr, (int, float)):
        errors.append(f"hr={hr!r} bukan angka")
    elif not (0 <= hr <= 300):
        errors.append(f"hr={hr} di luar range [0, 300] BPM")

    # 5. finger: bool
    finger = payload.get("finger", None)
    if not isinstance(finger, bool):
        errors.append(f"finger={finger!r} bukan bool")

    # 6. SpO2 (opsional): jika ada, range 0.0-100.0
    spo2 = payload.get("spo2", None)
    if spo2 is not None:
        if not isinstance(spo2, (int, float)):
            errors.append(f"spo2={spo2!r} bukan angka")
        elif not (0.0 <= float(spo2) <= 100.0):
            errors.append(f"spo2={spo2} di luar range [0, 100]%")

    # 7. Timestamp monotonicity
    if node_id >= 0 and isinstance(ts, (int, float)) and ts >= 0:
        errors += _check_ts_monotonic(node_id, "ppg", int(ts))

    ok = len(errors) == 0
    return ok, errors


def reset_node_state(node_id: int) -> None:
    """Reset timestamp state untuk node tertentu (misal: setelah reboot terdeteksi)."""
    for sig_type in ("imu", "ppg"):
        _last_ts.pop((node_id, sig_type), None)


def get_validation_stats() -> dict:
    """Return info state validator saat ini (untuk debugging)."""
    return {
        "tracked_nodes": len({k[0] for k in _last_ts}),
        "last_ts": {f"node{k[0]}_{k[1]}": v for k, v in _last_ts.items()},
        "whitelist": sorted(ALLOWED_NODE_IDS) if ALLOWED_NODE_IDS else "all",
    }
