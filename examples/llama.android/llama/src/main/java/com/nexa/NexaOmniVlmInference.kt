package com.nexa
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn

class NexaOmniVlmInference(
    private val modelPath: String,
    private val projectorPath: String,
    private var imagePath: String,
    private var stopWords: List<String> = emptyList(),
    private var temperature: Float = 0.8f,
    private var maxNewTokens: Int = 64,
    private var topK: Int = 40,
    private var topP: Float = 0.95f
) {
    init {
        System.loadLibrary("omni-android")
    }
    private var embed_imgage_pointer: Long = 0
    private var sampler_pointer: Long = 0
    private var nPastPointer: Long = 0
    private var generatedTokenNum: Int = 0
    private var generatedText: String = ""
    private var isModelLoaded: Boolean = false
    private var type:String = "vlm-81-ocr"

    private external fun init(model: String, proj: String, type: String)
    private external fun sampler_free(sampler:Long)
    private external fun free()

    private external fun image_embed(prompt:String, image:String): Long
    private external fun sampler_init( prompt: String, image: String, npast: Long, jimage_embed: Long): Long
    private external fun inference( npast: Long,  sampler:Long): String
    private external fun npast_init():Long

    @Synchronized
    fun loadModel() {
        if(isModelLoaded){
            throw RuntimeException("Model is already loaded.")
        }
        try {
            init(modelPath, projectorPath, type)
            isModelLoaded = true
        } catch (e: Exception) {
            println(e)
        } catch (e: UnsatisfiedLinkError) {
            throw RuntimeException("Native method not found: ${e.message}")
        }
    }

    fun dispose() {
        free()
    }

    private fun updateParams(
        stopWords: List<String>? = null,
        temperature: Float? = null,
        maxNewTokens: Int? = null,
        topK: Int? = null,
        topP: Float? = null
    ) {
        if(stopWords != null){
            this.stopWords = stopWords
        }
        if (temperature != null) {
            this.temperature = temperature
        }
        if (maxNewTokens != null) {
            this.maxNewTokens = maxNewTokens
        }
        if (topK != null) {
            this.topK = topK;
        }
        if (topP != null) {
            this.topP = topP
        }

    }

    private fun shouldStop(): Boolean {
        if(this.generatedTokenNum >= this.maxNewTokens){
            return true
        }

        return stopWords.any { generatedText.contains(it, ignoreCase = true) }
    }

    private fun resetGeneration() {
        generatedTokenNum = 0
        generatedText = ""
    }

    @Synchronized
    fun createCompletionStream(
        prompt: String,
        imagePath: String? = null,
        stopWords: List<String>? = null,
        temperature: Float? = null,
        maxNewTokens: Int? = null,
        topK: Int? = null,
        topP: Float? = null
    ): Flow<String> = flow {
        if(!isModelLoaded){
            throw RuntimeException("Model is not loaded.")
        }

        // Reset generation state at the start
        resetGeneration()
        updateParams(stopWords, temperature, maxNewTokens, topK, topP)
        val imagePathToUse = imagePath ?: this@NexaOmniVlmInference.imagePath
        nPastPointer = npast_init();
        embed_imgage_pointer =  image_embed(prompt, imagePathToUse)
        sampler_pointer = sampler_init(prompt, imagePathToUse, nPastPointer, embed_imgage_pointer)

        try {
            while (true) {
                val sampledText = inference(nPastPointer, sampler_pointer)
                generatedTokenNum += 1
                generatedText += sampledText
                if(shouldStop()){
                    break
                }
                emit(sampledText)
            }
        } finally {
            // Clean up resources and reset generation state
            resetGeneration()
            sampler_free(sampler_pointer)
        }

        println("This is a new thread!")
        // Your thread logic here
    }.flowOn(Dispatchers.IO)
}
