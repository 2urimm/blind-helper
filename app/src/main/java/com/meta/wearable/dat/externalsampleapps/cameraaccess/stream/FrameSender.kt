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

class FrameSender(private val serverUrl: String) {

    companion object {
        private const val TAG = "FrameSender"
        private const val JPEG_QUALITY = 60   // 낮을수록 대역폭 적게 먹음 (40~70 조절)
        private const val MAX_QUEUE_BYTES = 50_000L  // 송신 버퍼가 이 크기 넘으면 프레임 버림
    }

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
                // 서버가 돌려준 탐지 결과 JSON (지금은 로그만; 2단계 오버레이용)
                Log.d(TAG, "서버 응답: $text")
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

    fun close() {
        connected.set(false)
        webSocket?.close(1000, "종료")
        webSocket = null
    }
}