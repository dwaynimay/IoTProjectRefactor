// File: firmware/src/task_cs_sender.cpp

// =============================================================================
// task_cs_sender.cpp — Task CS Encode & Kirim dengan Dynamic Routing
// =============================================================================
//
// PERUBAHAN v3.0 (Multi-Hop):
//   - DynamicRouter digunakan sebelum kirim untuk menentukan rute
//   - DIRECT  : kirim langsung ke gateway (MacAddr::GATEWAY)
//   - RELAYED : kirim ke neighbor (MacAddr::NODE_A atau NODE_B)
//     Neighbor yang menerima akan memanggil forwardRoutedCs() secara otomatis
//     dari _onDataRecv di EspNowMesh (bukan dari task ini)
//
//   - taskRssiExchange() : task baru, kirim RssiReport ke neighbor
//     setiap RSSI_EXCHANGE_MS agar neighbor tahu RSSI kita
//
// CATATAN RELAY:
//   Saat node kirim ke neighbor (RELAYED), node TIDAK membungkus dalam
//   RoutedCsPacket — itu tugas relay node di _onDataRecv.
//   Node pengirim hanya kirim CS1AxisPacket/CSPpgPacket biasa ke neighbor MAC.
//   Relay node yang bungkus dan forward ke gateway.
// =============================================================================

#include <Arduino.h>
#include "Config.h"

#if NODE_ROLE == ROLE_SENSOR
#include "CS_Sensor.h"
#include "EspNowMesh.h"
#include "MeshPackets.h"
#include "Watchdog.h"
#include "DynamicRouter.h"

extern portMUX_TYPE g_stateMux;
extern ImuSample    g_latestImu;
extern PpgSample    g_latestPpg;
extern EspNowMesh   g_mesh;

static constexpr char TAG[] = "CS_TX";

// =============================================================================
// Encoder instances
// =============================================================================
static CSEncoder g_encAx, g_encAy, g_encAz;
static CSEncoder g_encGx, g_encGy, g_encGz;
static CSEncoder g_encIr;

// =============================================================================
// DynamicRouter — satu per sensor node
// =============================================================================
static DynamicRouter g_router(NODE_ID);

// Ekspos ke EspNowMesh agar _onDataRecv bisa update RSSI
DynamicRouter* g_routerPtr = &g_router;


// =============================================================================
// Helper: pilih MAC tujuan berdasarkan RouteDecision
// =============================================================================
static const uint8_t* _selectDstMac(const RouteDecision& dec)
{
    if (dec.isDirect)
        return MacAddr::GATEWAY;

    // Relay ke neighbor
    #if NODE_ID == 1
        return MacAddr::NODE_B;
    #else
        return MacAddr::NODE_A;
    #endif
}


// =============================================================================
// taskRssiExchange — Kirim RSSI ke neighbor setiap RSSI_EXCHANGE_MS
//
// Task ini berjalan di Core 0, prioritas rendah.
// Setiap RSSI_EXCHANGE_MS, baca RSSI terakhir dari beacon dan kirim ke neighbor
// via sendRssiReport(). Neighbor update DynamicRouter-nya dengan nilai ini.
// =============================================================================
void taskRssiExchange(void* param)
{
    static constexpr char RTAG[] = "RSSI_EX";
    uint32_t lastExchangeMs = 0;

    LOG_INFO(RTAG, "taskRssiExchange dimulai | interval=%lu ms",
             (unsigned long)RoutingCfg::RSSI_EXCHANGE_MS);

    // Tunggu discovery phase selesai dulu
    vTaskDelay(pdMS_TO_TICKS(RoutingCfg::DISCOVERY_PHASE_MS));

    for (;;)
    {
        vTaskDelay(pdMS_TO_TICKS(RoutingCfg::RSSI_EXCHANGE_MS));

        const int8_t myRssi = g_mesh.getLastBeaconRssi();

        if (myRssi == RoutingCfg::RSSI_UNKNOWN)
        {
            LOG_EVERY_N(5, LOG_WARN, RTAG,
                        "Belum terima beacon dari gateway — skip exchange");
            continue;
        }

        // Update router kita sendiri dengan RSSI self
        g_router.updateSelfRssi(myRssi);

        // Kirim ke neighbor
        const bool ok = g_mesh.sendRssiReport(NODE_ID, myRssi);

        LOG_DEBUG(RTAG, "RSSI exchange | self=%d dBm | ok=%s",
                  myRssi, ok ? "Y" : "N");

        // Print status routing setiap 10 detik
        LOG_EVERY_N(5, LOG_INFO, RTAG, "=== Status routing ===");
        if ((millis() / RoutingCfg::RSSI_EXCHANGE_MS) % 5 == 0)
            g_router.printStatus();
    }
}


// =============================================================================
// Sensor Sanity Check — batas fisis sensor
// =============================================================================
//
//  IMU (MPU6050 default config):
//    Accel range : ±2g  → ±19.62 m/s²  (pakai ±25 untuk margin)
//    Gyro range  : ±250 °/s             (pakai ±300 untuk margin)
//
//  PPG:
//    IR raw      : harus > 0 jika finger=true
//
//  Jika gagal, window TIDAK di-encode dan di-kirim.
//  Counter g_droppedWindows dilog setiap DROPPED_LOG_INTERVAL window.
// =============================================================================
namespace SanityLimit {
    // Accel: MPU6050 default ±2g → ±19.62 m/s², pakai ±25 sebagai batas keras
    static constexpr float ACCEL_MAX_MS2  = 25.0f;
    // Gyro: MPU6050 default ±250 °/s, pakai ±300 sebagai batas keras
    static constexpr float GYRO_MAX_DEGS  = 300.0f;
    // Log setiap N drop agar tidak spam
    static constexpr uint32_t DROPPED_LOG_INTERVAL = 10;
}

static bool _imuInRange(const ImuSample& s)
{
    if (fabsf(s.accelX) > SanityLimit::ACCEL_MAX_MS2) return false;
    if (fabsf(s.accelY) > SanityLimit::ACCEL_MAX_MS2) return false;
    if (fabsf(s.accelZ) > SanityLimit::ACCEL_MAX_MS2) return false;
    if (fabsf(s.gyroX)  > SanityLimit::GYRO_MAX_DEGS) return false;
    if (fabsf(s.gyroY)  > SanityLimit::GYRO_MAX_DEGS) return false;
    if (fabsf(s.gyroZ)  > SanityLimit::GYRO_MAX_DEGS) return false;
    return true;
}


// =============================================================================
// taskCSSender — Entry Point Task
// =============================================================================
void taskCSSender(void* param)
{
    g_watchdog.registerTask();

    float yAx[CS_M], yAy[CS_M], yAz[CS_M];
    float yGx[CS_M], yGy[CS_M], yGz[CS_M];
    float yIr[CS_M];

    uint32_t windowCount    = 0;
    uint32_t directCount    = 0;
    uint32_t relayedCount   = 0;
    uint32_t droppedWindows = 0;   // counter drop akibat sanity check

    CSPhiMatrix::printInfo();
    CSPhiMatrix::printSyncDebug();

    LOG_INFO(TAG, "7 encoder aktif | N=%d M=%d | Multi-hop routing AKTIF",
             CS_N, CS_M);
    LOG_INFO(TAG, "Discovery phase: %lu ms | RSSI threshold: %d dBm",
             (unsigned long)RoutingCfg::DISCOVERY_PHASE_MS,
             RoutingCfg::RELAY_THRESHOLD_DBM);

    for (;;)
    {
        g_watchdog.feed();

        // ── Snapshot shared state ─────────────────────────────────────────────
        ImuSample imu{};
        PpgSample ppg{};
        taskENTER_CRITICAL(&g_stateMux);
        imu = g_latestImu;
        ppg = g_latestPpg;
        taskEXIT_CRITICAL(&g_stateMux);

        // ── Sanity Check: validasi range fisis sensor ─────────────────────────
        const bool fingerDetected = (ppg.irRaw >= EdgeConfig::IR_FINGER_THRESHOLD);

        if (!_imuInRange(imu))
        {
            droppedWindows++;
            if (droppedWindows % SanityLimit::DROPPED_LOG_INTERVAL == 0)
            {
                LOG_WARN(TAG,
                         "SANITY DROP #%lu | IMU out-of-range "
                         "ax=%.2f ay=%.2f az=%.2f gx=%.2f gy=%.2f gz=%.2f",
                         droppedWindows,
                         imu.accelX, imu.accelY, imu.accelZ,
                         imu.gyroX,  imu.gyroY,  imu.gyroZ);
            }
            vTaskDelay(pdMS_TO_TICKS(Timing::IMU_SAMPLE_MS));
            continue;
        }

        // PPG: jika jari tidak terdeteksi, skip IR encoding (isi 0)
        // Sensor tetap di-push ke encoder IMU agar buffer tidak stall
        const float irSample = fingerDetected
                               ? static_cast<float>(ppg.irRaw)
                               : 0.0f;

        // ── Push sample ke semua encoder ──────────────────────────────────────
        const bool axRdy = g_encAx.pushSample(imu.accelX);
        const bool ayRdy = g_encAy.pushSample(imu.accelY);
        const bool azRdy = g_encAz.pushSample(imu.accelZ);
        const bool gxRdy = g_encGx.pushSample(imu.gyroX);
        const bool gyRdy = g_encGy.pushSample(imu.gyroY);
        const bool gzRdy = g_encGz.pushSample(imu.gyroZ);
        const bool irRdy = g_encIr.pushSample(irSample);

        if (!(axRdy && ayRdy && azRdy && gxRdy && gyRdy && gzRdy && irRdy))
        {
            vTaskDelay(pdMS_TO_TICKS(Timing::IMU_SAMPLE_MS));
            continue;
        }

        // ── Encode semua sinyal ───────────────────────────────────────────────
        g_encAx.encode(yAx);
        g_encAy.encode(yAy);
        g_encAz.encode(yAz);
        g_encGx.encode(yGx);
        g_encGy.encode(yGy);
        g_encGz.encode(yGz);
        g_encIr.encode(yIr);

        const bool     finger = fingerDetected;  // already computed above
        const uint32_t tsNow  = millis();

        // ── Dynamic Routing Decision ──────────────────────────────────────────
        const RouteDecision dec   = g_router.decide();
        const uint8_t* dstMac     = _selectDstMac(dec);

        if (dec.isDirect) directCount++;
        else              relayedCount++;

        // ── Kirim 6 IMU axis ke tujuan yang dipilih ───────────────────────────
        uint8_t nack = 0;

        if (!g_mesh.sendCsAxis(PKT_CS_AX, NODE_ID, yAx, finger, tsNow, dstMac)) nack++;
        if (!g_mesh.sendCsAxis(PKT_CS_AY, NODE_ID, yAy, finger, tsNow, dstMac)) nack++;
        if (!g_mesh.sendCsAxis(PKT_CS_AZ, NODE_ID, yAz, finger, tsNow, dstMac)) nack++;
        if (!g_mesh.sendCsAxis(PKT_CS_GX, NODE_ID, yGx, finger, tsNow, dstMac)) nack++;
        if (!g_mesh.sendCsAxis(PKT_CS_GY, NODE_ID, yGy, finger, tsNow, dstMac)) nack++;
        if (!g_mesh.sendCsAxis(PKT_CS_GZ, NODE_ID, yGz, finger, tsNow, dstMac)) nack++;
        if (!g_mesh.sendCsPpg(NODE_ID, yIr, ppg.heartRate,
                               ppg.valid, ppg.spo2,
                               finger, tsNow, dstMac))     nack++;

        windowCount++;

        // ── Log ───────────────────────────────────────────────────────────────
        if (windowCount % 5 == 0)
        {
            const char* routeStr = dec.isDirect ? "DIRECT" : "RELAY";
            const float relayPct = windowCount > 0
                                   ? 100.0f * relayedCount / windowCount
                                   : 0.0f;

            LOG_INFO(TAG,
                     "Win #%lu [%s] | self=%d dBm neighbor=%d dBm | "
                     "relay=%.0f%% | nack=%d | HR=%d SpO2=%.1f%%",
                     windowCount,
                     routeStr,
                     dec.rssiSelf,
                     dec.rssiNeighbor,
                     relayPct,
                     nack,
                     ppg.heartRate,
                     ppg.spo2 > 0 ? ppg.spo2 : 0.0f);
        }

        if (nack > 0)
        {
            LOG_WARN(TAG, "Window #%lu — %d/7 paket gagal TX ke %s!",
                     windowCount,
                     nack,
                     dec.isDirect ? "GATEWAY" : "RELAY");
        }

        LOG_EVERY_N(500, LOG_DEBUG, TAG,
                    "Stack: %u bytes | heap: %lu KB",
                    uxTaskGetStackHighWaterMark(NULL),
                    esp_get_free_heap_size() / 1024);

        vTaskDelay(pdMS_TO_TICKS(Timing::IMU_SAMPLE_MS));
    }
}

#endif // NODE_ROLE == ROLE_SENSOR
