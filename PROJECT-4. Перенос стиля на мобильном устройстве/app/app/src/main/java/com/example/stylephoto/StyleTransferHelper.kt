package com.example.stylephoto

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.io.FileInputStream
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class StyleTransferHelper(context: Context) {

    private val interpreter: Interpreter

    init {
        val model = loadModelFile(context, "style_transfer.tflite")

        interpreter = Interpreter(model)
    }

    private fun loadModelFile(context: Context, filename: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun stylize(content: Bitmap, style: Bitmap): Bitmap {
        val contentBuffer = bitmapToBuffer(content)
        val styleBuffer   = bitmapToBuffer(style)

        val output = Array(1) { Array(256) { Array(256) { FloatArray(3) } } }
        val outputs = hashMapOf<Int, Any>(0 to output)

        interpreter.runForMultipleInputsOutputs(
            arrayOf(styleBuffer, contentBuffer),
            outputs
        )

        return bufferToBitmap(output)
    }

    private fun bitmapToBuffer(bitmap: Bitmap): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, 256, 256, true)

        val buffer = ByteBuffer.allocateDirect(1 * 256 * 256 * 3 * 4)
        buffer.order(ByteOrder.nativeOrder())

        for (y in 0 until 256) {
            for (x in 0 until 256) {
                val pixel = resized.getPixel(x, y)

                buffer.putFloat(((pixel shr 16) and 0xFF) / 255f) // R
                buffer.putFloat(((pixel shr 8) and 0xFF) / 255f)  // G
                buffer.putFloat((pixel and 0xFF) / 255f)          // B
            }
        }

        buffer.rewind()

        return buffer
    }

    private fun bufferToBitmap(output: Array<Array<Array<FloatArray>>>): Bitmap {
        val bitmap = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888)

        for (y in 0 until 256) {
            for (x in 0 until 256) {
                val r = (output[0][y][x][0] * 255).toInt().coerceIn(0, 255)
                val g = (output[0][y][x][1] * 255).toInt().coerceIn(0, 255)
                val b = (output[0][y][x][2] * 255).toInt().coerceIn(0, 255)

                val color = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                bitmap.setPixel(x, y, color)
            }
        }

        return bitmap
    }
}