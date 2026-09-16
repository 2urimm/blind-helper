/*
 * InferenceDecoder - 추론 전용 HEVC 디코더 (경로 A, A-1)
 *
 * 프리뷰용 HevcDecoder와 별개로, 같은 압축 HEVC 프레임을 받아 ImageReader로 디코딩해
 * 비트맵을 뽑고 YoloTflite로 추론한다. 프리뷰(HevcDecoder)는 전혀 건드리지 않는다.
 *
 * 부하를 줄이기 위해:
 *   - 프레임을 매번 추론하지 않고, 이전 추론이 끝났을 때만 새 프레임을 추론(최신 프레임 우선).
 *   - HevcDecoder의 NAL 파싱/enqueue 로직을 그대로 재사용.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.stream

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import android.media.ImageReader
import android.media.MediaCodec
import android.media.MediaCodecList
import android.media.MediaFormat
import android.os.Handler
import android.os.HandlerThread
import android.os.Process
import android.util.Log
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import com.meta.wearable.dat.externalsampleapps.cameraaccess.camera.InferenceMode

class InferenceDecoder(
  context: Context,
  private val serverUrl: String,
  private val onDetections: (List<Detection>) -> Unit,
) {
  // 서버 모드에서 쓸 전송기 (지연 생성)
  private var frameSender: FrameSender? = null
  @Volatile var mode: InferenceMode = InferenceMode.ON_DEVICE

  companion object {
    private const val TAG = "InferenceDecoder"
    private const val DATA_QUEUE_CAPACITY = 100
    private val BLOCKED_DECODERS = setOf("OMX.Exynos.hevc.dec", "c2.mtk.hevc.decoder")
  }

  private val yolo = YoloTflite(context)
  // 추론 중이면 새 프레임 스킵 (밀림 방지)
  private val inferring = AtomicBoolean(false)
  private val inferThread = HandlerThread("YoloInferThread").apply { start() }
  private val inferHandler = Handler(inferThread.looper)

  private class DecoderFrame(
      val data: ByteBuffer,
      val offset: Int = 0,
      val size: Int = data.remaining(),
      val presentationTimeUs: Long = 0L,
      val isKeyFrame: Boolean = false,
      val isConfigFrame: Boolean = false,
  ) {
    val flags: Int
      get() {
        var bitmask = 0
        if (isKeyFrame) bitmask = bitmask or MediaCodec.BUFFER_FLAG_KEY_FRAME
        if (isConfigFrame) bitmask = bitmask or MediaCodec.BUFFER_FLAG_CODEC_CONFIG
        return bitmask
      }
  }

  @Volatile private var decoder: MediaCodec? = null
  @Volatile private var decoderThread: HandlerThread? = null
  private val incomingDataQueue = LinkedBlockingQueue<DecoderFrame>(DATA_QUEUE_CAPACITY)

  @Volatile private var mediaFormat: MediaFormat? = null
  @Volatile private var cachedVideoCodec: ByteBuffer? = null
  @Volatile private var active = false
  @Volatile private var firstInputFrame = true
  @Volatile private var receivedKeyframe = false
  @Volatile private var imageReader: ImageReader? = null

  fun start(width: Int, height: Int) {
    imageReader =
        ImageReader.newInstance(width, height, ImageFormat.YUV_420_888, 2).apply {
          setOnImageAvailableListener(
              { reader ->
                val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                try {
                  if (inferring.compareAndSet(false, true)) {
                    val bmp = yuvToBitmap(image)
                    inferHandler.post {
                      try {
                        if (bmp != null) {
                          when (mode) {
                            InferenceMode.ON_DEVICE -> {
                              val dets = yolo.detect(bmp)
                              onDetections(dets)
                            }
                            InferenceMode.SERVER -> {
                              frameSender?.sendFrameForMetrics(bmp)
                              onDetections(emptyList()) // 서버 모드는 폰 박스 안 그림(로그로 측정)
                            }
                            InferenceMode.OFF -> {}
                          }
                        }
                      } catch (e: Exception) {
                        Log.e(TAG, "처리 에러: ${e.message}")
                      } finally {
                        inferring.set(false)
                      }
                    }
                  }
                } catch (e: Exception) {
                  Log.e(TAG, "프레임 처리 에러: ${e.message}")
                  inferring.set(false)
                } finally {
                  image.close()
                }
              },
              Handler(HandlerThread("InferImageReader").apply { start() }.looper),
          )
        }

    mediaFormat =
        MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_HEVC, width, height).also { format ->
          format.setInteger(MediaFormat.KEY_FRAME_RATE, 30)
          format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
          format.setInteger(MediaFormat.KEY_BIT_RATE, 750000)
        }
    try {
      ensureCodecsCreated()
    } catch (e: Exception) {
      Log.e(TAG, "디코더 생성 실패: ${e.message}", e)
    }
  }

  fun decodeFrame(data: ByteArray, presentationTimeUs: Long) {
    if (data.isEmpty()) return
    val buffer = ByteBuffer.wrap(data)
    val writableByteArray = data.copyOf()
    var index = 0
    val prefixFlags = BooleanArray(3)

    index = findNalUnit(writableByteArray, index, data.size, prefixFlags)
    while (index < data.size) {
      val unitType = getH265NalUnitType(writableByteArray, index)
      val isKeyFrame = isIrapNalType(unitType)
      val isConfigFrame = unitType == 32 || unitType == 33 || unitType == 34

      if (isConfigFrame) {
        cachedVideoCodec = cloneByteBuffer(buffer)
      } else if (isKeyFrame) {
        if (!active) {
          active = true
          cachedVideoCodec?.let { enqueuePublic(it) }
        }
        if (!receivedKeyframe) receivedKeyframe = true
      }

      enqueuePrivate(
          DecoderFrame(
              data = cloneByteBuffer(buffer),
              offset = index,
              presentationTimeUs = presentationTimeUs,
              isKeyFrame = isKeyFrame,
              isConfigFrame = isConfigFrame,
          )
      )
      index = findNalUnit(writableByteArray, index + 1, data.size, prefixFlags)
    }
  }
  fun connectServer() {
    if (frameSender == null) {
      frameSender = FrameSender(serverUrl).also { it.connect() }
    }
  }

  fun disconnectServer() {
    frameSender?.close()
    frameSender = null
  }

  fun stop() {
    active = false
    incomingDataQueue.clear()
    try {
      decoder?.stop()
      decoder?.release()
    } catch (e: Exception) {
      Log.e(TAG, "디코더 정지 에러: ${e.message}")
    }
    decoder = null
    decoderThread?.quit()
    decoderThread = null
    firstInputFrame = true
    receivedKeyframe = false
    cachedVideoCodec = null
    imageReader?.close()
    imageReader = null
    inferThread.quitSafely()
    yolo.close()
    frameSender?.close()
    frameSender = null
  }

  private fun enqueuePublic(buffer: ByteBuffer) {
    val writableByteArray =
        ByteBuffer.allocate(buffer.capacity())
            .apply {
              buffer.rewind(); put(buffer); flip(); buffer.rewind()
            }
            .array()
    var index = 0
    val prefixFlags = BooleanArray(3)
    index = findNalUnit(writableByteArray, index, buffer.limit(), prefixFlags)
    while (index < buffer.limit()) {
      val unitType = getH265NalUnitType(writableByteArray, index)
      val isKeyFrame = isIrapNalType(unitType)
      val isConfigFrame = unitType == 32 || unitType == 33 || unitType == 34
      if (isConfigFrame) {
        cachedVideoCodec = cloneByteBuffer(buffer)
      } else if (isKeyFrame) {
        if (!receivedKeyframe) receivedKeyframe = true
      }
      enqueuePrivate(
          DecoderFrame(
              data = cloneByteBuffer(buffer),
              offset = index,
              presentationTimeUs = 0,
              isKeyFrame = isKeyFrame,
              isConfigFrame = isConfigFrame,
          )
      )
      index = findNalUnit(writableByteArray, index + 1, buffer.limit(), prefixFlags)
    }
  }

  private fun enqueuePrivate(frame: DecoderFrame) {
    if (!active) return
    if (!frame.isConfigFrame && !receivedKeyframe) return
    if (firstInputFrame) {
      firstInputFrame = false
      activateDecoder()
    }
    if (incomingDataQueue.remainingCapacity() == 0) {
      active = false
      return
    }
    incomingDataQueue.offer(frame)
  }

  private fun ensureCodecsCreated() {
    if (decoder == null) decoder = createHevcDecoder()
  }

  private fun createHevcDecoder(): MediaCodec {
    val mime = MediaFormat.MIMETYPE_VIDEO_HEVC
    val softwareName =
        MediaCodecList(MediaCodecList.ALL_CODECS)
            .codecInfos
            .firstOrNull { info ->
              !info.isEncoder &&
                  info.isSoftwareOnly &&
                  info.name !in BLOCKED_DECODERS &&
                  info.supportedTypes.any { it.equals(mime, ignoreCase = true) }
            }
            ?.name
    return if (softwareName != null) MediaCodec.createByCodecName(softwareName)
    else MediaCodec.createDecoderByType(mime)
  }

  private fun activateDecoder() {
    try {
      ensureCodecsCreated()
      decoder?.let { codec ->
        decoderThread?.quit()
        val thread = HandlerThread("InferDecoderThread", Process.THREAD_PRIORITY_VIDEO)
        thread.start()
        decoderThread = thread
        codec.reset()
        codec.configure(mediaFormat, imageReader?.surface, null, 0)
        codec.setCallback(
            object : MediaCodec.Callback() {
              override fun onInputBufferAvailable(codec: MediaCodec, index: Int) =
                  onInputBuffer(codec, index)

              override fun onOutputBufferAvailable(
                  codec: MediaCodec,
                  index: Int,
                  info: MediaCodec.BufferInfo,
              ) = onOutputBuffer(codec, index, info)

              override fun onError(codec: MediaCodec, e: MediaCodec.CodecException) {
                Log.e(TAG, "코덱 에러: ${e.message}")
              }

              override fun onOutputFormatChanged(codec: MediaCodec, format: MediaFormat) {}
            },
            Handler(thread.looper),
        )
        codec.start()
      }
    } catch (e: Throwable) {
      Log.e(TAG, "디코더 활성화 에러: ${e.message}", e)
    }
  }

  private fun onInputBuffer(codec: MediaCodec, index: Int) {
    var queued = false
    try {
      val inputBuffer = codec.getInputBuffer(index)
      val frame = incomingDataQueue.poll(1, TimeUnit.SECONDS)
      if (frame == null || inputBuffer == null || !active) {
        codec.queueInputBuffer(index, 0, 0, 0, 0)
        queued = true
        return
      }
      frame.data.rewind()
      inputBuffer.clear()
      inputBuffer.put(frame.data)
      inputBuffer.flip()
      val clampedSize = minOf(frame.size, inputBuffer.limit() - frame.offset)
      codec.queueInputBuffer(index, frame.offset, clampedSize, frame.presentationTimeUs, frame.flags)
      queued = true
    } catch (e: Throwable) {
      if (active) active = false
    } finally {
      if (!queued) {
        try {
          codec.queueInputBuffer(index, 0, 0, 0, 0)
        } catch (_: Throwable) {}
      }
    }
  }

  private fun onOutputBuffer(codec: MediaCodec, index: Int, info: MediaCodec.BufferInfo) {
    try {
      if (!active || info.size == 0) {
        codec.releaseOutputBuffer(index, false)
        return
      }
      // ImageReader Surface로 렌더 → onImageAvailable 발생
      codec.releaseOutputBuffer(index, true)
    } catch (e: Throwable) {
      try {
        codec.releaseOutputBuffer(index, false)
      } catch (_: Throwable) {}
    }
  }

  // --- YUV_420_888 → Bitmap ---
  private fun yuvToBitmap(image: Image): Bitmap? {
    val nv21 = yuv420ToNv21(image)
    val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 85, out)
    val bytes = out.toByteArray()
    return android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
  }

  private fun yuv420ToNv21(image: Image): ByteArray {
    val width = image.width
    val height = image.height
    val ySize = width * height
    val uvSize = width * height / 4
    val nv21 = ByteArray(ySize + uvSize * 2)
    val yBuffer = image.planes[0].buffer
    val uBuffer = image.planes[1].buffer
    val vBuffer = image.planes[2].buffer
    var rowStride = image.planes[0].rowStride
    var pos = 0
    if (rowStride == width) {
      yBuffer.get(nv21, 0, ySize); pos += ySize
    } else {
      var yPos = 0
      while (pos < ySize) {
        yBuffer.position(yPos); yBuffer.get(nv21, pos, width); yPos += rowStride; pos += width
      }
    }
    rowStride = image.planes[2].rowStride
    val pixelStride = image.planes[2].pixelStride
    val uvHeight = height / 2
    val uvWidth = width / 2
    for (row in 0 until uvHeight) {
      for (col in 0 until uvWidth) {
        val vuPos = col * pixelStride + row * rowStride
        nv21[pos++] = vBuffer.get(vuPos)
        nv21[pos++] = uBuffer.get(vuPos)
      }
    }
    return nv21
  }

  // --- NAL 파싱 (HevcDecoder와 동일) ---
  private fun findNalUnit(data: ByteArray, startOffset: Int, endOffset: Int, prefixFlags: BooleanArray): Int {
    val length = endOffset - startOffset
    if (length == 0) return endOffset
    when {
      prefixFlags[0] -> { clearPrefixFlags(prefixFlags); return startOffset - 3 }
      length > 1 && prefixFlags[1] && data[startOffset].toInt() == 1 -> { clearPrefixFlags(prefixFlags); return startOffset - 2 }
      length > 2 && prefixFlags[2] && data[startOffset].toInt() == 0 && data[startOffset + 1].toInt() == 1 -> { clearPrefixFlags(prefixFlags); return startOffset - 1 }
    }
    val limit = endOffset - 1
    var i = startOffset + 2
    while (i < limit) {
      if ((data[i].toInt() and 0xFE) != 0) {
      } else if (data[i - 2].toInt() == 0 && data[i - 1].toInt() == 0 && data[i].toInt() == 1) {
        clearPrefixFlags(prefixFlags); return i - 2
      } else { i -= 2 }
      i += 3
    }
    prefixFlags[0] =
        if (length > 2) (data[endOffset - 3].toInt() == 0 && data[endOffset - 2].toInt() == 0 && data[endOffset - 1].toInt() == 1)
        else (if (length == 2) (prefixFlags[2] && data[endOffset - 2].toInt() == 0 && data[endOffset - 1].toInt() == 1) else (prefixFlags[1] && data[endOffset - 1].toInt() == 1))
    prefixFlags[1] =
        if (length > 1) (data[endOffset - 2].toInt() == 0 && data[endOffset - 1].toInt() == 0)
        else (prefixFlags[2] && data[endOffset - 1].toInt() == 0)
    prefixFlags[2] = data[endOffset - 1].toInt() == 0
    return endOffset
  }

  private fun clearPrefixFlags(prefixFlags: BooleanArray) {
    prefixFlags[0] = false; prefixFlags[1] = false; prefixFlags[2] = false
  }

  private fun getH265NalUnitType(data: ByteArray, offset: Int): Int {
    if (offset + 3 >= data.size) return -1
    return (data[offset + 3].toInt() and 0x7E) shr 1
  }

  private fun isIrapNalType(unitType: Int): Boolean = unitType in 16..21

  private fun cloneByteBuffer(original: ByteBuffer): ByteBuffer {
    val clone = if (original.isDirect) ByteBuffer.allocateDirect(original.capacity()) else ByteBuffer.allocate(original.capacity())
    original.rewind(); clone.put(original); original.rewind(); clone.flip()
    return clone
  }
}
