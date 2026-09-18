/*
 * ObstacleAlarm - 시각장애인 보조 장애물 알람
 *
 * 서버가 준 detections(박스+거리+라벨)로:
 *   1) 방향 판정 (좌/정면/우) — 박스 중심 x
 *   2) 정면 최근접 물체의 거리 시계열 → 접근속도 → TTC(충돌예상시간)
 *   3) 3단계 알람: 위험/경고/주의 → 비프(빠르기 차등) + 진동 + TTS
 *
 * C방식: 평소 비프로 거리감, 위험할 때 TTS로 "방향 거리 종류".
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.stream

import android.content.Context
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.Locale

class ObstacleAlarm(context: Context) {

  companion object {
    private const val TAG = "ObstacleAlarm"
    // 거리 임계 (m)
    private const val ALARM_MAX_DIST = 4.0f    // 이보다 멀면 알람 안 함
    private const val DANGER_DIST = 0.8f       // 이보다 가까우면 무조건 위험
    // TTC 임계 (초)
    private const val TTC_DANGER = 2.5f
    private const val TTC_WARN = 4.5f
    // 정면 영역 폭 (가로의 가운데 비율). 0.33 = 가운데 1/3
    private const val FRONT_RATIO = 0.6f
    // 알람 간격
    private const val BEEP_COOLDOWN_MS = 250L
    private const val TTS_COOLDOWN_MS = 2500L
    // 거리 시계열 유지 개수
    private const val HISTORY = 6
  }

  private var ttsEngine: TextToSpeech? = null
  private var ttsReady = false

  private val tone = ToneGenerator(AudioManager.STREAM_MUSIC, 90)
  private val vibrator: Vibrator? =
      context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator

  // 정면 최근접 거리 시계열: (시각ms, 거리m)
  private val distHistory = ArrayDeque<Pair<Long, Float>>()

  private var lastBeepMs = 0L
  private var lastTtsMs = 0L
  private var lastLevel = 0   // 0 없음, 1 주의, 2 경고, 3 위험

  private var lastSeenMs = 0L

  init {
    ttsEngine = TextToSpeech(context) { status ->
      if (status == TextToSpeech.SUCCESS) {
        ttsEngine?.language = Locale.KOREAN
        ttsReady = true
      } else {
        Log.w(TAG, "TTS 초기화 실패")
      }
    }
  }

  /** 서버 결과로 알람 판정. frameW = 서버 프레임 가로(방향 계산 기준). */
  fun process(dets: List<Detection>, frameW: Int) {
    val now = System.currentTimeMillis()
    val w = if (frameW > 0) frameW.toFloat() else 504f

    // 정면 영역 [left, right]
    val frontHalf = w * FRONT_RATIO / 2f
    val cx0 = w / 2f - frontHalf
    val cx1 = w / 2f + frontHalf

    // 정면 & 거리 있는 물체 중 최근접 하나
    var nearest: Detection? = null
    for (d in dets) {
      val dist = d.dist ?: continue
      if (dist <= 0f || dist > ALARM_MAX_DIST) continue
      val cx = (d.left + d.right) / 2f
      if (cx < cx0 || cx > cx1) continue   // 정면만
      if (nearest == null || dist < (nearest!!.dist ?: Float.MAX_VALUE)) {
        nearest = d
      }
    }

    if (nearest == null) {
      // 잠깐 놓친 것과 진짜 없어진 것 구분: 마지막 감지 후 시간 체크
      if (now - lastSeenMs > 800) {
        distHistory.clear()
        lastLevel = 0
      }
      return
    }
    lastSeenMs = now

    val dist = nearest.dist!!
    // 거리 시계열 갱신
    distHistory.addLast(now to dist)
    while (distHistory.size > HISTORY) distHistory.removeFirst()

    // 접근속도 추정 (가장 오래된 것 vs 최신, m/s). 양수 = 가까워지는 중
    val ttc = estimateTtc(dist)

    // 알람 레벨 판정
    val level = when {
      dist <= DANGER_DIST -> 3
      ttc != null && ttc <= TTC_DANGER -> 3
      ttc != null && ttc <= TTC_WARN -> 2
      else -> 1
    }

    // 방향
    val cx = (nearest.left + nearest.right) / 2f
    val dir = when {
      cx < w / 3f -> "왼쪽"
      cx > w * 2f / 3f -> "오른쪽"
      else -> "정면"
    }

    fireAlarm(level, dir, dist, nearest.label, now)
  }

  /** 거리 시계열로 TTC(초) 추정. 접근 안 하면 null. */
  private fun estimateTtc(curDist: Float): Float? {
    if (distHistory.size < 2) return null
    val (t0, d0) = distHistory.first()
    val (t1, d1) = distHistory.last()
    val dtSec = (t1 - t0) / 1000f
    if (dtSec <= 0f) return null
    val approach = (d0 - d1) / dtSec   // m/s, 양수면 가까워지는 중
    if (approach <= 0.05f) return null // 멀어지거나 정지 → 충돌 위험 없음
    return curDist / approach
  }

  private fun fireAlarm(level: Int, dir: String, dist: Float, label: String, now: Long) {
    // 비프: 레벨별 빠르기
    val beepInterval = when (level) {
      3 -> 0L      // 위험: 거의 연속
      2 -> 300L    // 경고: 빠름
      else -> 800L // 주의: 느림
    }
    if (now - lastBeepMs >= beepInterval + BEEP_COOLDOWN_MS) {
      val toneType = when (level) {
        3 -> ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD
        2 -> ToneGenerator.TONE_PROP_BEEP2
        else -> ToneGenerator.TONE_PROP_BEEP
      }
      tone.startTone(toneType, if (level == 3) 200 else 120)
      lastBeepMs = now
    }

    // 위험: 진동 + TTS
    if (level >= 2) {
      if (level == 3) vibrate(300)
      else if (lastLevel < 2) vibrate(120)
      // 경고·위험 모두 TTS로 뭔지 알려줌 (쿨다운 내)
      if (now - lastTtsMs >= TTS_COOLDOWN_MS && ttsReady) {
        val distText = "%.1f".format(dist)
        ttsEngine?.speak("$dir, ${distText}미터, $label", TextToSpeech.QUEUE_FLUSH, null, "obstacle")
        lastTtsMs = now
      }
    }
    lastLevel = level
  }

  private fun vibrate(ms: Long) {
    val v = vibrator ?: return
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
      v.vibrate(VibrationEffect.createOneShot(ms, VibrationEffect.DEFAULT_AMPLITUDE))
    } else {
      @Suppress("DEPRECATION") v.vibrate(ms)
    }
  }

  fun release() {
    try { tone.release() } catch (_: Exception) {}
    try { ttsEngine?.stop(); ttsEngine?.shutdown() } catch (_: Exception) {}
  }
}
