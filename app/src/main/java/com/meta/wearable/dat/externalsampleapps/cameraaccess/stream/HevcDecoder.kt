/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// HevcDecoder - 원본 복구본 (Surface 직행 프리뷰) + 측정 로그 유지
// 경로 A(온디바이스)로 전환하며 프리뷰를 되살리기 위해 ImageReader/FrameSender 제거

package com.meta.wearable.dat.externalsampleapps.cameraaccess.stream

import android.media.MediaCodec
import android.media.MediaCodecList
import android.media.MediaFormat
import android.os.Handler
import android.os.HandlerThread
import android.os.Process
import android.util.Log
import android.view.Surface
import java.nio.ByteBuffer
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

class HevcDecoder {

  companion object {
    private const val TAG = "HevcDecoder"
    private const val DATA_QUEUE_CAPACITY = 100
    private val BLOCKED_DECODERS = setOf("OMX.Exynos.hevc.dec", "c2.mtk.hevc.decoder")
  }

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

  // --- 측정용 카운터 ---
  private var inputFrameCount = 0
  private var lastInputLogMs = 0L
  private var outputFrameCount = 0
  private var lastOutputLogMs = 0L

  @Volatile private var decoder: MediaCodec? = null
  @Volatile private var decoderThread: HandlerThread? = null
  private val incomingDataQueue = LinkedBlockingQueue<DecoderFrame>(DATA_QUEUE_CAPACITY)

  @Volatile private var mediaFormat: MediaFormat? = null
  @Volatile private var cachedVideoCodec: ByteBuffer? = null
  @Volatile private var active = false
  @Volatile private var firstInputFrame = true
  @Volatile private var receivedKeyframe = false
  @Volatile private var outputSurface: Surface? = null

  fun start(width: Int, height: Int, surface: Surface) {
    Log.d("STREAM_METRICS", "resolution=${width}x${height}")
    outputSurface = surface
    mediaFormat =
      MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_HEVC, width, height).also { format
        ->
        format.setInteger(MediaFormat.KEY_FRAME_RATE, 30)
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        format.setInteger(MediaFormat.KEY_BIT_RATE, 750000)
      }
    try {
      ensureCodecsCreated()
    } catch (e: Exception) {
      Log.e(TAG, "Failed to create HEVC decoder: ${e.message}", e)
    }
  }

  fun decodeFrame(data: ByteArray, presentationTimeUs: Long) {
    if (data.isEmpty()) return

    // --- 측정용 ---
    inputFrameCount++
    val nowMs = System.currentTimeMillis()
    if (lastInputLogMs == 0L) lastInputLogMs = nowMs
    if (nowMs - lastInputLogMs >= 1000) {
      Log.d("STREAM_METRICS", "input_fps=$inputFrameCount  ptsUs=$presentationTimeUs")
      inputFrameCount = 0
      lastInputLogMs = nowMs
    }
    // -------------

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
          cachedVideoCodec?.let { cachedConfig -> enqueuePublic(cachedConfig) }
        }
        if (!receivedKeyframe) {
          receivedKeyframe = true
        }
      }

      enqueuePrivate(
        DecoderFrame(
          data = cloneByteBuffer(buffer),
          offset = index,
          presentationTimeUs = presentationTimeUs,
          isKeyFrame = isKeyFrame,
          isConfigFrame = isConfigFrame,
        ),
      )

      index = findNalUnit(writableByteArray, index + 1, data.size, prefixFlags)
    }
  }

  fun stop() {
    active = false
    incomingDataQueue.clear()
    try {
      decoder?.stop()
      decoder?.release()
    } catch (e: Exception) {
      Log.e(TAG, "Error stopping decoder: ${e.message}", e)
    }
    decoder = null
    decoderThread?.quit()
    decoderThread = null
    firstInputFrame = true
    receivedKeyframe = false
    cachedVideoCodec = null
    outputSurface = null
  }

  private fun enqueuePublic(buffer: ByteBuffer) {
    val writableByteArray =
      ByteBuffer.allocate(buffer.capacity())
        .apply {
          buffer.rewind()
          put(buffer)
          flip()
          buffer.rewind()
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
        if (!receivedKeyframe) {
          receivedKeyframe = true
        }
      }
      enqueuePrivate(
        DecoderFrame(
          data = cloneByteBuffer(buffer),
          offset = index,
          presentationTimeUs = 0,
          isKeyFrame = isKeyFrame,
          isConfigFrame = isConfigFrame,
        ),
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
      Log.w(TAG, "Decoder queue full")
      active = false
      return
    }
    incomingDataQueue.offer(frame)
  }

  private fun ensureCodecsCreated() {
    if (decoder == null) {
      decoder = createHevcDecoder()
    }
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
    return if (softwareName != null) {
      Log.d(TAG, "Using software HEVC decoder: $softwareName")
      MediaCodec.createByCodecName(softwareName)
    } else {
      Log.w(TAG, "No software HEVC decoder found; using platform default")
      MediaCodec.createDecoderByType(mime)
    }
  }

  private fun activateDecoder() {
    try {
      ensureCodecsCreated()
      decoder?.let { codec ->
        decoderThread?.quit()
        val thread = HandlerThread("HevcDecoderThread", Process.THREAD_PRIORITY_VIDEO)
        thread.start()
        decoderThread = thread

        codec.reset()
        codec.configure(mediaFormat, outputSurface, null, 0)
        codec.setCallback(
          object : MediaCodec.Callback() {
            override fun onInputBufferAvailable(codec: MediaCodec, index: Int) {
              onInputBuffer(codec, index)
            }

            override fun onOutputBufferAvailable(
              codec: MediaCodec,
              index: Int,
              info: MediaCodec.BufferInfo,
            ) {
              onOutputBuffer(codec, index, info)
            }

            override fun onError(codec: MediaCodec, e: MediaCodec.CodecException) {
              Log.e(TAG, "Codec error: ${e.message}")
            }

            override fun onOutputFormatChanged(codec: MediaCodec, format: MediaFormat) {}
          },
          Handler(thread.looper),
        )
        codec.start()
      }
    } catch (e: MediaCodec.CodecException) {
      Log.e(TAG, "Decoder activation codec exception: ${e.message}", e)
    } catch (e: Throwable) {
      Log.e(TAG, "Decoder activation exception: ${e.message}", e)
    }
  }

  private fun onInputBuffer(codec: MediaCodec, index: Int) {
    var bufferQueued = false
    try {
      val inputBuffer = codec.getInputBuffer(index)
      val frame = incomingDataQueue.poll(1, TimeUnit.SECONDS)

      if (frame == null || inputBuffer == null || !active) {
        codec.queueInputBuffer(index, 0, 0, 0, 0)
        bufferQueued = true
        return
      }

      frame.data.rewind()
      inputBuffer.clear()
      inputBuffer.put(frame.data)
      inputBuffer.flip()
      val clampedSize = minOf(frame.size, inputBuffer.limit() - frame.offset)
      codec.queueInputBuffer(
        index,
        frame.offset,
        clampedSize,
        frame.presentationTimeUs,
        frame.flags,
      )
      bufferQueued = true
    } catch (e: Throwable) {
      Log.e(TAG, "Input buffer error: ${e.message}", e)
      if (active) active = false
    } finally {
      if (!bufferQueued) {
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
      // --- 측정용 ---
      outputFrameCount++
      val outNowMs = System.currentTimeMillis()
      if (lastOutputLogMs == 0L) lastOutputLogMs = outNowMs
      if (outNowMs - lastOutputLogMs >= 1000) {
        Log.d("STREAM_METRICS", "decoded_fps=$outputFrameCount")
        outputFrameCount = 0
        lastOutputLogMs = outNowMs
      }
      // -------------
      // Surface로 직접 렌더 → 폰 화면에 프리뷰 표시
      codec.releaseOutputBuffer(index, true)
    } catch (e: Throwable) {
      Log.e(TAG, "Output buffer error: ${e.message}", e)
      try {
        codec.releaseOutputBuffer(index, false)
      } catch (_: Throwable) {}
    }
  }

  // --- NalUnitUtil (copied from SDK) ---

  private fun findNalUnit(
    data: ByteArray,
    startOffset: Int,
    endOffset: Int,
    prefixFlags: BooleanArray,
  ): Int {
    val length = endOffset - startOffset
    if (length == 0) return endOffset

    when {
      prefixFlags[0] -> {
        clearPrefixFlags(prefixFlags)
        return startOffset - 3
      }
      length > 1 && prefixFlags[1] && data[startOffset].toInt() == 1 -> {
        clearPrefixFlags(prefixFlags)
        return startOffset - 2
      }
      length > 2 &&
              prefixFlags[2] &&
              data[startOffset].toInt() == 0 &&
              data[startOffset + 1].toInt() == 1 -> {
        clearPrefixFlags(prefixFlags)
        return startOffset - 1
      }
    }

    val limit = endOffset - 1
    var i = startOffset + 2
    while (i < limit) {
      if ((data[i].toInt() and 0xFE) != 0) {
        // no NAL prefix here or next two positions
      } else if (data[i - 2].toInt() == 0 && data[i - 1].toInt() == 0 && data[i].toInt() == 1) {
        clearPrefixFlags(prefixFlags)
        return i - 2
      } else {
        i -= 2
      }
      i += 3
    }

    prefixFlags[0] =
      if (length > 2)
        (data[endOffset - 3].toInt() == 0 &&
                data[endOffset - 2].toInt() == 0 &&
                data[endOffset - 1].toInt() == 1)
      else
        (if (length == 2)
          (prefixFlags[2] &&
                  data[endOffset - 2].toInt() == 0 &&
                  data[endOffset - 1].toInt() == 1)
        else (prefixFlags[1] && data[endOffset - 1].toInt() == 1))
    prefixFlags[1] =
      if (length > 1) (data[endOffset - 2].toInt() == 0 && data[endOffset - 1].toInt() == 0)
      else (prefixFlags[2] && data[endOffset - 1].toInt() == 0)
    prefixFlags[2] = data[endOffset - 1].toInt() == 0

    return endOffset
  }

  private fun clearPrefixFlags(prefixFlags: BooleanArray) {
    prefixFlags[0] = false
    prefixFlags[1] = false
    prefixFlags[2] = false
  }

  private fun getH265NalUnitType(data: ByteArray, offset: Int): Int {
    if (offset + 3 >= data.size) return -1
    return (data[offset + 3].toInt() and 0x7E) shr 1
  }

  private fun isIrapNalType(unitType: Int): Boolean = unitType in 16..21

  private fun cloneByteBuffer(original: ByteBuffer): ByteBuffer {
    val clone: ByteBuffer =
      if (original.isDirect) ByteBuffer.allocateDirect(original.capacity())
      else ByteBuffer.allocate(original.capacity())
    original.rewind()
    clone.put(original)
    original.rewind()
    clone.flip()
    return clone
  }
}