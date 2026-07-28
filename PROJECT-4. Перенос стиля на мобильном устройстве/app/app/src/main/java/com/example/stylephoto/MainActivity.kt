package com.example.stylephoto

import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import java.io.File

class MainActivity : AppCompatActivity() {

    private val REQUEST_IMAGE_CAPTURE = 1
    private val CAMERA_PERMISSION_CODE = 100

    private lateinit var styleHelper: StyleTransferHelper
    private lateinit var imageView: ImageView
    private lateinit var progressBar: ProgressBar
    private lateinit var stylesRecyclerView: RecyclerView
    private lateinit var btnTakePhoto: Button
    private lateinit var btnSave: Button
    private lateinit var tvHint: TextView

    private var capturedBitmap: Bitmap? = null
    private var styledBitmap: Bitmap? = null
    private var currentPhotoPath: String = ""
    private var styleFileNames: List<String> = emptyList()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        styleHelper       = StyleTransferHelper(this)
        imageView         = findViewById(R.id.imageViewResult)
        progressBar       = findViewById(R.id.progressBar)
        stylesRecyclerView= findViewById(R.id.stylesRecyclerView)
        btnTakePhoto      = findViewById(R.id.btnTakePhoto)
        btnSave            = findViewById(R.id.btnSave)
        tvHint            = findViewById(R.id.tvHint)

        styleFileNames = assets.list("styles")
            ?.filter { it.endsWith(".jpg") || it.endsWith(".png") }
            ?: emptyList()

        if (styleFileNames.isEmpty()) {
            Toast.makeText(this, "No style images found in assets/styles/", Toast.LENGTH_LONG).show()
        }

        setupStylePicker()

        btnTakePhoto.setOnClickListener { checkCameraPermission() }

        btnSave.setOnClickListener {
            styledBitmap?.let { saveImageToGallery(it) }
                ?: Toast.makeText(this, "No styled image to save", Toast.LENGTH_SHORT).show()
        }
    }

    // ── Style picker ──────────────────────────────────────────────────────────

    private fun setupStylePicker() {
        val thumbnails = styleFileNames.map { name ->
            BitmapFactory.decodeStream(assets.open("styles/$name"))
        }

        val adapter = StyleAdapter(thumbnails) { index -> applyStyle(index) }
        stylesRecyclerView.layoutManager =
            LinearLayoutManager(this, LinearLayoutManager.HORIZONTAL, false)
        stylesRecyclerView.adapter = adapter
    }

    private fun applyStyle(styleIndex: Int) {
        val content = capturedBitmap ?: run {
            Toast.makeText(this, "Take a photo first!", Toast.LENGTH_SHORT).show()
            return
        }

        val styleBitmap = BitmapFactory.decodeStream(
            assets.open("styles/${styleFileNames[styleIndex]}")
        )

        progressBar.visibility = View.VISIBLE
        btnTakePhoto.isEnabled = false
        btnSave.visibility = View.GONE
        tvHint.text = "Applying style…"

        Thread {
            val result = styleHelper.stylize(content, styleBitmap)
            runOnUiThread {
                progressBar.visibility = View.GONE
                btnTakePhoto.isEnabled = true
                tvHint.text = "Pick a style or retake photo"
                imageView.setImageBitmap(result)
                styledBitmap = result
                btnSave.visibility = View.VISIBLE
            }
        }.start()
    }

    // ── Camera ────────────────────────────────────────────────────────────────

    private fun checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            launchCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(android.Manifest.permission.CAMERA),
                CAMERA_PERMISSION_CODE
            )
        }
    }

    private fun launchCamera() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        if (intent.resolveActivity(packageManager) == null) {
            Toast.makeText(this, "No camera app found", Toast.LENGTH_LONG).show()
            return
        }
        val photoFile = createImageFile()
        val uri = FileProvider.getUriForFile(
            this, "com.example.stylephoto.fileprovider", photoFile
        )
        intent.putExtra(MediaStore.EXTRA_OUTPUT, uri)
        startActivityForResult(intent, REQUEST_IMAGE_CAPTURE)
    }

    private fun createImageFile(): File {
        val stamp = System.currentTimeMillis()
        val dir = getExternalFilesDir("Pictures")
        return File.createTempFile("JPEG_${stamp}_", ".jpg", dir).also {
            currentPhotoPath = it.absolutePath
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                launchCamera()
            } else {
                Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            try {
                capturedBitmap = BitmapFactory.decodeFile(currentPhotoPath)
                imageView.setImageBitmap(capturedBitmap)
                tvHint.text = "Now pick a style below ↓"
                btnSave.visibility = View.GONE
                styledBitmap = null
            } catch (e: Exception) {
                Toast.makeText(this, "Error loading photo: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // ── Save ──────────────────────────────────────────────────────────────────

    private fun saveImageToGallery(bitmap: Bitmap) {
        try {
            val url = MediaStore.Images.Media.insertImage(
                contentResolver,
                bitmap,
                "StylePhoto_${System.currentTimeMillis()}",
                "Generated with Style Transfer"
            )
            val msg = if (url != null) "✅ Saved to Gallery!" else "❌ Failed to save"
            Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
        } catch (e: Exception) {
            Toast.makeText(this, "Error saving: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }
}