
package com.ethicalml.kompute

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import android.widget.Toast
import android.view.View
import android.widget.EditText
import android.widget.TextView
import com.ethicalml.kompute.databinding.ActivityKomputeJniBinding

class KomputeJni : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val binding = ActivityKomputeJniBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.komputeGifView.loadUrl("file:///android_asset/komputer-2.gif")

        binding.komputeGifView.getSettings().setUseWideViewPort(true)
        binding.komputeGifView.getSettings().setLoadWithOverviewMode(true)
        
        binding.predictionTextView.text = "N/A"
    }

    fun KomputeButtonOnClick(v: View) {

        val xiEditText = findViewById<EditText>(R.id.XIEditText)
        val xjEditText = findViewById<EditText>(R.id.XJEditText)
        val yEditText = findViewById<EditText>(R.id.YEditText)

        val wOneEditText = findViewById<TextView>(R.id.wOneTextView)
        val wTwoEditText = findViewById<TextView>(R.id.wTwoTextView)
        val biasEditText = findViewById<TextView>(R.id.biasTextView)

        val komputeJniTextview = findViewById<TextView>(R.id.predictionTextView)

        val xi = xiEditText.text.removeSurrounding("[", "]").split(",").map { it.toFloat() }.toFloatArray()
        val xj = xjEditText.text.removeSurrounding("[", "]").split(",").map { it.toFloat() }.toFloatArray()
        val y = yEditText.text.removeSurrounding("[", "]").split(",").map { it.toFloat() }.toFloatArray()

        val out = kompute(xi, xj, y)

        Log.i("KomputeJni", "RESULT:")
        Log.i("KomputeJni", out.contentToString())

        komputeJniTextview.text = out.contentToString()

        val params = komputeParams(xi, xj, y)

        Log.i("KomputeJni", "Params:")
        Log.i("KomputeJni", params.contentToString())

        wOneEditText.text = params[0].toString()
        wTwoEditText.text = params[1].toString()
        biasEditText.text = params[2].toString()
    }

    external fun initVulkan(): Boolean

    external fun kompute(xi: FloatArray, xj: FloatArray, y: FloatArray): FloatArray

    external fun komputeParams(xi: FloatArray, xj: FloatArray, y: FloatArray): FloatArray

    companion object {
        init {
            System.loadLibrary("kompute-jni")
        }
    }
}

