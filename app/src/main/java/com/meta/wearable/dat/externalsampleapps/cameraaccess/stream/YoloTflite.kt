/*
 * YoloTflite - 온디바이스 YOLO11n 추론 (경로 A)
 *
 * assets의 yolo11n_float16.tflite 로드 → 비트맵을 320x320으로 전처리 → 추론
 * → 출력 [1,84,2100] 파싱 → NMS → 탐지 결과 반환.
 *
 * 모델 출력 형식 (ultralytics YOLO11 export 기준):
 *   [1, 84, 2100]  = [batch, 4박스+80클래스, 후보 2100개]  (transposed)
 *   - 채널 0~3   : cx, cy, w, h  (0~1 정규화된 좌표, 320 기준)
 *   - 채널 4~83  : 80개 COCO 클래스 점수 (sigmoid 이미 적용됨 / 0~1)
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.stream

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

data class Detection(
    val label: String,
    val score: Float,
    // 박스 좌표 (입력 320x320 기준 픽셀). 화면 오버레이(A-2) 때 원본 크기로 스케일링.
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float,
)

class YoloTflite(context: Context) {

  companion object {
    private const val TAG = "YoloTflite"
    private const val MODEL_FILE = "yolo11n_float16.tflite"
    private const val LABEL_FILE = "coco_labels.txt"
    private const val INPUT_SIZE = 320
    private const val NUM_CLASSES = 80
    private const val NUM_CHANNELS = 84      // 4 + 80
    private const val NUM_BOXES = 2100
    private const val CONF_THRES = 0.4f      // 이 점수 미만 무시
    private const val IOU_THRES = 0.45f      // NMS 겹침 임계값
    private const val NUM_THREADS = 4

    // fps 측정용
    private var frameCount = 0
    private var lastFpsLogMs = 0L
    private var currentFps = 0.0
  }

  private val interpreter: Interpreter
  private val labels: List<String>

  // 재사용 버퍼 (매 프레임 할당 방지)
  private val inputBuffer: ByteBuffer =
      ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4).apply {
        order(ByteOrder.nativeOrder())
      }
  private val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
  private val output = Array(1) { Array(NUM_CHANNELS) { FloatArray(NUM_BOXES) } }

  init {
    val opts = Interpreter.Options().apply { numThreads = NUM_THREADS }
    interpreter = Interpreter(loadModelFile(context), opts)
    labels = context.assets.open(LABEL_FILE).bufferedReader().readLines()
    Log.d(TAG, "모델 로드 완료. 라벨 ${labels.size}개")
  }

  private fun loadModelFile(context: Context): ByteBuffer {
    val fd = context.assets.openFd(MODEL_FILE)
    FileInputStream(fd.fileDescriptor).use { input ->
      val channel = input.channel
      return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }
  }

  /** 비트맵 한 장 추론 → 탐지 목록. 추론시간(ms)은 로그로 찍음. */
  fun detect(bitmap: Bitmap): List<Detection> {
    val t0 = System.currentTimeMillis()

    // 1) 전처리: 320x320 리사이즈 → 정규화(0~1) → inputBuffer 채우기
    val resized =
        if (bitmap.width == INPUT_SIZE && bitmap.height == INPUT_SIZE) bitmap
        else Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)

    resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
    inputBuffer.rewind()
    for (p in pixels) {
      // ARGB → R,G,B 순서, 0~1 정규화
      inputBuffer.putFloat(((p shr 16) and 0xFF) / 255f)
      inputBuffer.putFloat(((p shr 8) and 0xFF) / 255f)
      inputBuffer.putFloat((p and 0xFF) / 255f)
    }

    // 2) 추론
    interpreter.run(inputBuffer, output)

    // 3) 출력 파싱: [1,84,2100] → 후보별 최고 클래스 골라 임계값 통과분만
    val candidates = ArrayList<Detection>()
    val out = output[0]  // [84][2100]
    for (i in 0 until NUM_BOXES) {
      // 최고 점수 클래스 찾기
      var bestCls = -1
      var bestScore = 0f
      for (c in 0 until NUM_CLASSES) {
        val s = out[4 + c][i]
        if (s > bestScore) {
          bestScore = s
          bestCls = c
        }
      }
      if (bestScore < CONF_THRES || bestCls < 0) continue

      // 박스 좌표 (cx,cy,w,h 는 0~1 → 320 픽셀로)
      val cx = out[0][i] * INPUT_SIZE
      val cy = out[1][i] * INPUT_SIZE
      val w = out[2][i] * INPUT_SIZE
      val h = out[3][i] * INPUT_SIZE
      val left = cx - w / 2
      val top = cy - h / 2
      val right = cx + w / 2
      val bottom = cy + h / 2

      candidates.add(
          Detection(labels.getOrElse(bestCls) { "cls$bestCls" }, bestScore, left, top, right, bottom)
      )
    }

    // 4) NMS (겹치는 박스 제거)
    val result = nms(candidates)

    val ms = System.currentTimeMillis() - t0

    // fps 계산 (1초마다 갱신)
    frameCount++
    val now = System.currentTimeMillis()
    if (lastFpsLogMs == 0L) lastFpsLogMs = now
    if (now - lastFpsLogMs >= 1000) {
      currentFps = frameCount * 1000.0 / (now - lastFpsLogMs)
      Log.d("YOLO_FPS", "fps=${"%.1f".format(currentFps)}  (추론 평균 ${ms}ms)")
      frameCount = 0
      lastFpsLogMs = now
    }

    if (result.isNotEmpty()) {
      val summary = result.joinToString(", ") { "${it.label} ${"%.2f".format(it.score)}" }
      Log.d("YOLO_RESULT", "$summary  (추론 ${ms}ms, 후보 ${candidates.size})")
    } else {
      Log.d("YOLO_RESULT", "탐지 없음  (추론 ${ms}ms)")
    }
    return result
  }

  private fun nms(dets: List<Detection>): List<Detection> {
    val sorted = dets.sortedByDescending { it.score }.toMutableList()
    val keep = ArrayList<Detection>()
    while (sorted.isNotEmpty()) {
      val best = sorted.removeAt(0)
      keep.add(best)
      sorted.removeAll { iou(best, it) > IOU_THRES }
    }
    return keep
  }

  private fun iou(a: Detection, b: Detection): Float {
    val x1 = max(a.left, b.left)
    val y1 = max(a.top, b.top)
    val x2 = min(a.right, b.right)
    val y2 = min(a.bottom, b.bottom)
    val inter = max(0f, x2 - x1) * max(0f, y2 - y1)
    val areaA = (a.right - a.left) * (a.bottom - a.top)
    val areaB = (b.right - b.left) * (b.bottom - b.top)
    val union = areaA + areaB - inter
    return if (union <= 0f) 0f else inter / union
  }

  fun close() {
    interpreter.close()
  }
}
