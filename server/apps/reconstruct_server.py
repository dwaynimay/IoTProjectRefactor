# File: server/apps/reconstruct_server.py

"""
reconstruct_server.py — Server rekonstruksi CS (Hybrid Topic)

Subscribe 2 topic per node:
  health_monitor/node_N/cs_imu  → rekonstruksi ax,ay,az,gx,gy,gz sekaligus
  health_monitor/node_N/cs_ppg  → rekonstruksi ir + metadata HR

Jalankan dari folder server/:
    python -m apps.reconstruct_server
"""

import json
import time
import threading
import warnings

import paho.mqtt.client as mqtt
try:
    from paho.mqtt.enums import CallbackAPIVersion
    _PAHO_V2 = True
except ImportError:
    _PAHO_V2 = False

from core.config import (
    CS_N, CS_M, MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE,
    TOPIC_BASE, SIGNALS, IMU_SIGNALS, PPG_SIGNALS,
    UNITS, TS_SPREAD_TOLERANCE_MS,
)
from core.cs_router import reconstruct, PHI
from core.validator import validate_imu, validate_ppg, reset_node_state
from core.quality import assess_window, window_summary


# =============================================================================
# NodeState — buffer per node, tunggu cs_imu DAN cs_ppg
# =============================================================================
class NodeState:
    def __init__(self, node_id: int):
        self.node_id       = node_id
        self._imu_buf      = None   # payload cs_imu terakhir
        self._ppg_buf      = None   # payload cs_ppg terakhir
        self._lock         = threading.Lock()
        self.windows_done  = 0
        self._last_win_t   = 0.0
        self._total_rec_ms = 0.0
        self._val_errors   = 0      # jumlah payload ditolak validator
        self._low_quality  = 0      # jumlah window LOW_QUALITY

    def on_imu(self, payload: dict):
        """Dipanggil saat cs_imu diterima — validasi dulu sebelum buffer."""
        ok, errors = validate_imu(payload, node_id=self.node_id)
        if not ok:
            print(f"[Node {self.node_id}] VALIDATION ERROR (cs_imu): "
                  f"{'; '.join(errors)}")
            self._val_errors += 1
            return
        with self._lock:
            self._imu_buf = payload
            self._try_reconstruct()

    def on_ppg(self, payload: dict):
        """Dipanggil saat cs_ppg diterima — validasi dulu sebelum buffer."""
        ok, errors = validate_ppg(payload, node_id=self.node_id)
        if not ok:
            print(f"[Node {self.node_id}] VALIDATION ERROR (cs_ppg): "
                  f"{'; '.join(errors)}")
            self._val_errors += 1
            return
        with self._lock:
            self._ppg_buf = payload
            self._try_reconstruct()

    def _try_reconstruct(self):
        """Rekonstruksi hanya jika kedua buffer sudah terisi."""
        if self._imu_buf is None or self._ppg_buf is None:
            return

        # Cek timestamp spread antara cs_imu dan cs_ppg
        ts_imu = self._imu_buf.get("ts", 0)
        ts_ppg = self._ppg_buf.get("ts", 0)
        spread = abs(ts_imu - ts_ppg)

        if spread > TS_SPREAD_TOLERANCE_MS:
            # Timestamp terlalu jauh — kemungkinan dari window berbeda
            # Buang yang lebih lama, tunggu pasangannya
            if ts_imu < ts_ppg:
                print(f"[Node {self.node_id}] WARN: cs_imu ts={ts_imu} "
                      f"terlalu lama vs cs_ppg ts={ts_ppg} "
                      f"(spread={spread}ms) — reset imu buf")
                self._imu_buf = None
            else:
                print(f"[Node {self.node_id}] WARN: cs_ppg ts={ts_ppg} "
                      f"terlalu lama vs cs_imu ts={ts_imu} "
                      f"(spread={spread}ms) — reset ppg buf")
                self._ppg_buf = None
            return

        # Ambil payload lalu reset buffer
        imu_data = self._imu_buf
        ppg_data = self._ppg_buf
        self._imu_buf = None
        self._ppg_buf = None

        # Jalankan rekonstruksi
        self._reconstruct(imu_data, ppg_data)

    def _reconstruct(self, imu_data: dict, ppg_data: dict):
        """Rekonstruksi semua 7 sinyal dari payload hybrid."""
        results      = {}
        measurements = {}   # simpan y vector untuk quality assessment
        t0 = time.time()

        # Rekonstruksi 6 sinyal IMU dari cs_imu
        for sig in IMU_SIGNALS:
            y = imu_data.get(sig, [])
            if len(y) == CS_M:
                results[sig]      = reconstruct(y)
                measurements[sig] = y
            else:
                print(f"[Node {self.node_id}] WARN: {sig} len={len(y)}, "
                      f"expected {CS_M}")

        # Rekonstruksi IR dari cs_ppg
        y_ir = ppg_data.get("ir", [])
        if len(y_ir) == CS_M:
            results["ir"]      = reconstruct(y_ir)
            measurements["ir"] = y_ir
        else:
            print(f"[Node {self.node_id}] WARN: ir len={len(y_ir)}, "
                  f"expected {CS_M}")

        elapsed_ms = (time.time() - t0) * 1000

        self.windows_done  += 1
        self._total_rec_ms += elapsed_ms
        now = time.time()
        gap_ms = (now - self._last_win_t) * 1000 if self._last_win_t else 0
        self._last_win_t = now

        # Metadata dari cs_ppg
        hr     = ppg_data.get("hr", -1)
        spo2   = ppg_data.get("spo2", None)
        finger = ppg_data.get("finger", False)
        ts     = imu_data.get("ts", 0)
        avg_ms = self._total_rec_ms / self.windows_done

        # ── Quality Assessment ─────────────────────────────────────────────────
        quality_map = assess_window(measurements, results, PHI)
        q_summary   = window_summary(quality_map)
        if q_summary["any_low_quality"]:
            self._low_quality += 1

        # ── Print Header ──────────────────────────────────────────────────────
        q_tag = "⚠ LOW_Q" if q_summary["any_low_quality"] else "OK"
        spo2_str = f" | SpO2={spo2:.1f}%" if spo2 is not None else ""
        print(f"\n[Node {self.node_id}] Window #{self.windows_done} "
              f"| ts={ts}ms | gap={gap_ms:.0f}ms "
              f"| HR={hr}{spo2_str} | finger={'Y' if finger else 'N'} "
              f"| rekon={elapsed_ms:.1f}ms | avg={avg_ms:.1f}ms "
              f"| quality={q_tag} "
              f"| val_err={self._val_errors} | low_q={self._low_quality}")

        # ── Print rekonstruksi + quality per sinyal ───────────────────────────
        for sig in IMU_SIGNALS:
            if sig in results:
                x    = results[sig]
                unit = UNITS[sig]
                q    = quality_map.get(sig)
                q_info = f" rel_err={q.relative_error:.3f}" if q else ""
                print(f"  {sig}: [{x.min():.3f} … {x.max():.3f}] {unit}{q_info}")

        if "ir" in results:
            x = results["ir"]
            q = quality_map.get("ir")
            q_info = f" rel_err={q.relative_error:.3f}" if q else ""
            print(f"  ir: [{x.min():.0f} … {x.max():.0f}] ADC{q_info}")

        # Print quality warnings jika ada
        for w in q_summary.get("warnings", []):
            print(f"  [QUALITY WARN] {w}")

        # ── Hook: storage / ML model (Phase 2 & 3) ────────────────────────────
        # Phase 2: storage.save_window(self.node_id, ts, results, q_summary)
        # Phase 3: ml_inference.predict(results, hr=hr, spo2=spo2)


# =============================================================================
# MQTT
# =============================================================================
_nodes: dict = {}

def _get_node(node_id: int) -> NodeState:
    if node_id not in _nodes:
        _nodes[node_id] = NodeState(node_id)
        print(f"[INFO] Node {node_id} terdaftar")
    return _nodes[node_id]

def _on_message(client, userdata, message):
    try:
        payload = json.loads(message.payload.decode())
    except Exception as e:
        print(f"[ERROR] JSON parse: {e}")
        return

    # Parse topic: health_monitor/node_1/cs_imu
    parts = message.topic.split("/")
    if len(parts) < 3:
        return

    try:
        node_id = int(parts[1].split("_")[1])
    except (IndexError, ValueError):
        return

    sig_type = parts[2]  # "cs_imu" atau "cs_ppg"
    node     = _get_node(node_id)

    if sig_type == "cs_imu":
        node.on_imu(payload)
    elif sig_type == "cs_ppg":
        node.on_ppg(payload)

def _on_connect(client, userdata, flags, rc, properties=None):
    rc_val = rc if isinstance(rc, int) else rc.value
    if rc_val == 0:
        print(f"[MQTT] Terhubung ke {MQTT_BROKER}:{MQTT_PORT}")
        # Subscribe 2 topic per node, wildcard + untuk semua node
        for topic_type in ["cs_imu", "cs_ppg"]:
            topic = f"{TOPIC_BASE}/+/{topic_type}"
            client.subscribe(topic)
            print(f"[MQTT] Subscribe: {topic}")
    else:
        print(f"[MQTT] Gagal rc={rc_val}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("=" * 55)
    print("  CS Reconstruction Server (Hybrid Topic)")
    print(f"  N={CS_N} M={CS_M} ({CS_M*100//CS_N}%) | OMP K=20")
    print(f"  Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"  Subscribe: {TOPIC_BASE}/+/cs_imu")
    print(f"           : {TOPIC_BASE}/+/cs_ppg")
    print(f"  TS tolerance: {TS_SPREAD_TOLERANCE_MS}ms")
    print("=" * 55)

    if _PAHO_V2:
        client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2)
    else:
        client = mqtt.Client()

    client.on_connect = _on_connect
    client.on_message = _on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=MQTT_KEEPALIVE)
    except Exception as e:
        print(f"\n[ERROR] Tidak bisa konek ke {MQTT_BROKER}:{MQTT_PORT}")
        print(f"  → {e}")
        exit(1)

    print("\nMenunggu data dari sensor node...\n")
    client.loop_forever()