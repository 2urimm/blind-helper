/*
 * FrameSender - 디코딩된 프레임을 JPEG로 압축해 노트북 서버로 WebSocket 전송
 * 경로 B (서버 릴레이) 실험용
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.stream

import android.graphics.Bitmap
import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import okio.ByteString.Companion.toByteString
import java.io.ByteArrayOutputStream
import java.util.concurrent.atomic.AtomicBoolean

class FrameSender(
    private val serverUrl: String,
    private val onServerResult: (List<Detection>, Int, Int) -> Unit = { _, _, _ -> },
) {
    companion object {
        private const val TAG = "FrameSender"
        private const val JPEG_QUALITY = 60   // 낮을수록 대역폭 적게 먹음 (40~70 조절)
        private const val MAX_QUEUE_BYTES = 50_000L  // 송신 버퍼가 이 크기 넘으면 프레임 버림
    }
    // 구간 측정용
    @Volatile private var sentAtMs = 0L
    private var netCount = 0
    private var netSumRtt = 0.0
    private var netSumServer = 0.0
    private var lastNetLogMs = 0L

    @Volatile private var lastEncodeMs = 0L
    private val client = OkHttpClient()
    private var webSocket: WebSocket? = null
    private val connected = AtomicBoolean(false)
    // 이전 프레임 전송이 안 끝났으면 새 프레임을 버림 (밀리지 않게)
    private val sending = AtomicBoolean(false)

    fun connect() {
        val request = Request.Builder().url(serverUrl).build()
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: okhttp3.Response) {
                connected.set(true)
                Log.d(TAG, "서버 연결됨: $serverUrl")
            }

            override fun onMessage(ws: WebSocket, text: String) {
                val rtt = System.currentTimeMillis() - sentAtMs
                // 서버가 보낸 server_ms 파싱 (간단 파싱)
                val serverMs = Regex("\"server_ms\":\\s*([0-9.]+)")
                    .find(text)?.groupValues?.get(1)?.toDoubleOrNull() ?: 0.0
                val network = rtt - serverMs

                netCount++
                netSumRtt += rtt
                netSumServer += serverMs
                val now = System.currentTimeMillis()
                if (lastNetLogMs == 0L) lastNetLogMs = now
                if (now - lastNetLogMs >= 1000) {
                    val avgRtt = netSumRtt / netCount
                    val avgServer = netSumServer / netCount
                    val avgNet = avgRtt - avgServer
                    Log.d("NET_METRICS",
                        "rtt=${"%.0f".format(avgRtt)}ms  server=${"%.0f".format(avgServer)}ms  " +
                                "network=${"%.0f".format(avgNet)}ms  encode=${lastEncodeMs}ms  → 병목:${if (avgNet > avgServer) "네트워크" else "서버추론"}")
                    netCount = 0; netSumRtt = 0.0; netSumServer = 0.0
                    lastNetLogMs = now
                }
                // 박스+거리 파싱 → 오버레이용 콜백
                try {
                    val fw = Regex("\"frame_w\":\\s*(\\d+)").find(text)?.groupValues?.get(1)?.toIntOrNull() ?: 0
                    val fh = Regex("\"frame_h\":\\s*(\\d+)").find(text)?.groupValues?.get(1)?.toIntOrNull() ?: 0
                    onServerResult(parseDets(text), fw, fh)
                } catch (e: Exception) {
                    Log.e(TAG, "파싱 에러: ${e.message}")
                }
            }

            override fun onFailure(ws: WebSocket, t: Throwable, response: okhttp3.Response?) {
                connected.set(false)
                Log.e(TAG, "연결 실패: ${t.message}")
            }

            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                connected.set(false)
                Log.d(TAG, "연결 종료: $reason")
            }
        })
    }

    /** 비트맵을 JPEG로 압축해서 전송. 이전 전송이 진행 중이면 이 프레임은 버림. */
    fun sendFrame(bitmap: Bitmap) {
        if (!connected.get()) return
        if (!sending.compareAndSet(false, true)) return  // 아직 전송 중이면 스킵

        try {
            val baos = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, baos)
            val bytes = baos.toByteArray()
            webSocket?.send(bytes.toByteString())
        } catch (e: Exception) {
            Log.e(TAG, "전송 에러: ${e.message}")
        } finally {
            sending.set(false)
        }
    }

    /** JPEG 바이트 전송. 송신 버퍼가 밀려있으면(이전 프레임이 아직 안 나감) 이 프레임은 버림. */
    fun sendJpeg(jpeg: ByteArray) {
        if (!connected.get()) return
        val ws = webSocket ?: return
        // ★ 핵심: OkHttp가 아직 못 보낸 양이 임계값 넘으면 스킵 → 큐 누적 방지
        if (ws.queueSize() > MAX_QUEUE_BYTES) {
            return
        }
        try {
            ws.send(jpeg.toByteString())
        } catch (e: Exception) {
            Log.e(TAG, "전송 에러: ${e.message}")
        }
    }

    /** 비트맵을 JPEG로 압축해 전송 (서버 모드용). queueSize로 밀림 방지. */
    fun sendFrameForMetrics(bitmap: android.graphics.Bitmap) {
        if (!connected.get()) return
        val ws = webSocket ?: return
        if (ws.queueSize() > MAX_QUEUE_BYTES) return
        try {
            val e0 = System.currentTimeMillis()
            val baos = java.io.ByteArrayOutputStream()
            bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, JPEG_QUALITY, baos)
            val encodeMs = System.currentTimeMillis() - e0
            sentAtMs = System.currentTimeMillis()
            lastEncodeMs = encodeMs
            ws.send(baos.toByteArray().toByteString())
        } catch (e: Exception) {
            Log.e(TAG, "전송 에러: ${e.message}")
        }
    }

    private fun parseDets(text: String): List<Detection> {
        val result = mutableListOf<Detection>()
        val detsBlock = Regex("\"dets\":\\s*\\[(.*)\\]", RegexOption.DOT_MATCHES_ALL)
            .find(text)?.groupValues?.get(1) ?: return result
        for (m in Regex("\\{[^}]*\\}").findAll(detsBlock)) {
            val o = m.value
            val label = Regex("\"label\":\\s*\"([^\"]*)\"").find(o)?.groupValues?.get(1) ?: continue
            val conf = Regex("\"conf\":\\s*([0-9.]+)").find(o)?.groupValues?.get(1)?.toFloatOrNull() ?: 0f
            val box = Regex("\"box\":\\s*\\[([^\\]]*)\\]").find(o)?.groupValues?.get(1) ?: continue
            val nums = box.split(",").mapNotNull { it.trim().toFloatOrNull() }
            if (nums.size < 4) continue
            val dist = Regex("\"dist\":\\s*([0-9.]+)").find(o)?.groupValues?.get(1)?.toFloatOrNull()
            result.add(Detection(label, conf, nums[0], nums[1], nums[2], nums[3], dist))
        }
        return result
    }

    fun close() {
        connected.set(false)
        webSocket?.close(1000, "종료")
        webSocket = null
    }
}