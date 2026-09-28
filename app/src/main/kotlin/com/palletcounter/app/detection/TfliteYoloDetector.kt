package com.palletcounter.app.detection

import android.os.SystemClock
import android.util.Log
import com.palletcounter.app.data.Accelerator
import com.palletcounter.core.detection.DetectionFrame
import com.palletcounter.core.detection.DetectorInfo
import com.palletcounter.core.detection.ModelSidecar
import com.palletcounter.core.detection.PalletDetector
import com.palletcounter.core.detection.YoloDecoder
import com.palletcounter.core.detection.YoloLayout
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer

/**
 * YOLO-family detector running on LiteRT (TensorFlow Lite).
 *
 * Supports float32 / int8 / uint8 inputs and outputs, NHWC or NCHW inputs, and every output
 * layout handled by the core [YoloDecoder]. Must be created, used and closed on one thread
 * (the GPU delegate is bound to the thread that created it).
 */
class TfliteYoloDetector(
    model: MappedByteBuffer,
    private val sidecar: ModelSidecar,
    accelerator: Accelerator,
    cpuThreads: Int,
    decoderThreshold: Float,
) : PalletDetector<FrameInput> {

    private var gpuDelegate: GpuDelegate? = null
    private val interpreter: Interpreter
    private val preprocessor: Preprocessor
    private val outputShape: IntArray
    private val outputType: DataType
    private val outputScale: Float
    private val outputZeroPoint: Int
    private val outputBuffer: ByteBuffer
    private val outputFloats: FloatArray
    private val decoder: YoloDecoder
    private val acceleratorLabel: String
    override val info: DetectorInfo

    init {
        val (interp, label) = createInterpreter(model, accelerator, cpuThreads)
        interpreter = interp
        acceleratorLabel = label
        try {
            val parts = inspectModel()
            preprocessor = parts.preprocessor
            outputShape = parts.outputShape
            outputType = parts.outputType
            outputScale = parts.outputScale
            outputZeroPoint = parts.outputZeroPoint
            outputBuffer = parts.outputBuffer
            outputFloats = parts.outputFloats
            decoder = YoloDecoder(sidecar.toDecoderConfig(confidenceThreshold = decoderThreshold))
            info = parts.info
        } catch (t: Throwable) {
            // Unsupported model: release the native interpreter/delegate before failing.
            close()
            throw t
        }
    }

    private class ModelParts(
        val preprocessor: Preprocessor,
        val outputShape: IntArray,
        val outputType: DataType,
        val outputScale: Float,
        val outputZeroPoint: Int,
        val outputBuffer: ByteBuffer,
        val outputFloats: FloatArray,
        val info: DetectorInfo,
    )

    private fun inspectModel(): ModelParts {
        val input = interpreter.getInputTensor(0)
        val shape = input.shape()
        require(shape.size == 4 && shape[0] == 1) { "Unsupported input shape ${shape.contentToString()}" }
        val channelsFirst = shape[1] == 3 && shape[3] != 3
        val inH = if (channelsFirst) shape[2] else shape[1]
        val inW = if (channelsFirst) shape[3] else shape[2]
        val inType = when (input.dataType()) {
            DataType.FLOAT32 -> InputType.FLOAT32
            DataType.UINT8 -> InputType.UINT8
            DataType.INT8 -> InputType.INT8
            else -> throw IllegalArgumentException("Unsupported input type ${input.dataType()}")
        }
        val inQuant = input.quantizationParams()
        val preprocessor = Preprocessor(
            inputWidth = inW,
            inputHeight = inH,
            type = inType,
            channelsFirst = channelsFirst,
            padValue = sidecar.letterboxPadValue,
            normalize = sidecar.inputNormalization != "none",
            quantScale = if (inQuant.scale != 0f) inQuant.scale else 1f,
            quantZeroPoint = inQuant.zeroPoint,
        )
        val output = interpreter.getOutputTensor(0)
        val outputShape = output.shape()
        val outputType = output.dataType()
        val layout = YoloLayout.resolve(outputShape, inW, inH, sidecar.decoderFormat())
        val info = DetectorInfo(
            name = sidecar.name,
            kind = "tflite-yolo",
            inputWidth = inW,
            inputHeight = inH,
            accelerator = acceleratorLabel,
            isSimulation = false,
            details = mapOf(
                "output" to outputShape.contentToString(),
                "layout" to layout.format.name,
                "input_type" to inType.name,
                "version" to sidecar.version,
            ),
        )
        Log.i(TAG, "Loaded ${sidecar.name}: input ${shape.contentToString()} $inType, output ${outputShape.contentToString()} $outputType, $acceleratorLabel")
        return ModelParts(
            preprocessor = preprocessor,
            outputShape = outputShape,
            outputType = outputType,
            outputScale = output.quantizationParams().scale,
            outputZeroPoint = output.quantizationParams().zeroPoint,
            outputBuffer = ByteBuffer.allocateDirect(output.numBytes()).order(ByteOrder.nativeOrder()),
            outputFloats = FloatArray(output.numElements()),
            info = info,
        )
    }

    private fun createInterpreter(model: MappedByteBuffer, accelerator: Accelerator, threads: Int): Pair<Interpreter, String> {
        if (accelerator != Accelerator.CPU) {
            var delegate: GpuDelegate? = null
            try {
                CompatibilityList().use { compat ->
                    if (accelerator == Accelerator.GPU || compat.isDelegateSupportedOnThisDevice) {
                        val d = GpuDelegate(compat.bestOptionsForThisDevice)
                        delegate = d
                        val interp = Interpreter(model, Interpreter.Options().addDelegate(d))
                        gpuDelegate = d
                        return interp to "GPU"
                    }
                }
            } catch (t: Throwable) {
                Log.w(TAG, "GPU delegate unavailable, falling back to CPU", t)
                runCatching { delegate?.close() }
                gpuDelegate = null
            }
        }
        val options = Interpreter.Options().setNumThreads(threads)
        return Interpreter(model, options) to "CPU ×$threads"
    }

    override fun detect(frame: FrameInput, timestampNanos: Long): DetectionFrame {
        val lb = preprocessor.letterboxFor(frame)
        val input = preprocessor.process(frame, lb)
        outputBuffer.rewind()
        val t0 = SystemClock.elapsedRealtimeNanos()
        interpreter.run(input, outputBuffer)
        val t1 = SystemClock.elapsedRealtimeNanos()
        readOutput()
        val detections = decoder.decode(outputFloats, outputShape, lb)
        return DetectionFrame(
            timestampNanos = timestampNanos,
            detections = detections,
            frameWidth = frame.uprightWidth,
            frameHeight = frame.uprightHeight,
            inferenceMillis = (t1 - t0) / 1e6f,
        )
    }

    private fun readOutput() {
        outputBuffer.rewind()
        when (outputType) {
            DataType.FLOAT32 -> outputBuffer.asFloatBuffer().get(outputFloats)
            DataType.UINT8 -> for (i in outputFloats.indices) {
                outputFloats[i] = ((outputBuffer.get().toInt() and 0xFF) - outputZeroPoint) * outputScale
            }
            DataType.INT8 -> for (i in outputFloats.indices) {
                outputFloats[i] = (outputBuffer.get().toInt() - outputZeroPoint) * outputScale
            }
            else -> throw IllegalStateException("Unsupported output type $outputType")
        }
    }

    override fun close() {
        interpreter.close()
        gpuDelegate?.close()
        gpuDelegate = null
    }

    private companion object {
        const val TAG = "TfliteYoloDetector"
    }
}
