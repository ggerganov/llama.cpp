package com.nexa
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn

class NexaVlmInference(
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
        System.loadLibrary("llama-android")
    }

    private var paramsPointer: Long = 0
    private var modelPointer: Long = 0
    private var llavaCtxPointer: Long = 0
    private var embedImagePointer: Long = 0
    private var samplerPointer: Long = 0
    private var nPastPointer: Long = 0
    private var generatedTokenNum: Int = 0
    private var generatedText: String = ""
    private var isModelLoaded: Boolean = false
    private var cachedTokenPointer: Long = 0

    private external fun init_params(modelPath: String, mmprojPath: String): Long

    private external fun update_params(params: Long, temperature: Float, topK: Int, topP: Float)

    private external fun load_model(params: Long): Long

    private external fun free_model(model: Long)

    private external fun llava_init_context(params: Long, model: Long): Long

    private external fun llava_ctx_free(ctx: Long)

    private external fun load_image(ctx: Long, params: Long, imagepath: String): Long

    private external fun llava_image_embed_free(llava_image_embed: Long)

    private external fun llava_eval(ctx: Long, params: Long, llava_image_embed: Long, prompt: String): Long

    private external fun llava_sampler_init(ctx: Long, params: Long): Long

    private external fun llava_sample(ctx: Long, sampler: Long, n_past: Long, cached_tokens: Long): String

    private external fun cached_token_init(): Long

    private external fun cached_token_free(cached_tokens: Long)

    private external fun llava_sample_free(sampler: Long)

    @Synchronized
    fun loadModel() {
        if(isModelLoaded){
            throw RuntimeException("Model is already loaded.")
        }
        try {
            paramsPointer = init_params(modelPath, mmprojPath = projectorPath)
            modelPointer = load_model(paramsPointer)
            isModelLoaded = true
        } catch (e: Exception) {
            println(e)
        } catch (e: UnsatisfiedLinkError) {
            throw RuntimeException("Native method not found: ${e.message}")
        }
    }

    fun dispose() {
        if(paramsPointer!=0L){
            paramsPointer = 0;
        }
        if (modelPointer != 0L) {
            free_model(modelPointer)
            modelPointer = 0;
        }
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

        if(paramsPointer != 0L) {
            update_params(paramsPointer, this.temperature, this.topK, this.topP)
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

//        val thread = Thread {

            val imagePathToUse = imagePath ?: this@NexaVlmInference.imagePath
            llavaCtxPointer = llava_init_context(paramsPointer, modelPointer)
            embedImagePointer = load_image(llavaCtxPointer, paramsPointer, imagePathToUse)
            nPastPointer = llava_eval(llavaCtxPointer, paramsPointer, embedImagePointer, prompt)
            samplerPointer = llava_sampler_init(llavaCtxPointer, paramsPointer)
            cachedTokenPointer = cached_token_init()

            try {
                while (true) {
                    val sampledText = llava_sample(llavaCtxPointer, samplerPointer, nPastPointer, cachedTokenPointer)
                    generatedTokenNum += 1
                    generatedText += sampledText
                    if(shouldStop()){
                        break
                    }
                    emit(sampledText)
                    print(sampledText)
                }
            } finally {
                // Clean up resources and reset generation state
                cleanupResources()
                resetGeneration()
            }

            println("This is a new thread!")
            // Your thread logic here
//        }
//        thread.start()
    }.flowOn(Dispatchers.IO)

    private fun cleanupResources() {
        if(cachedTokenPointer != 0L){
            cached_token_free(cachedTokenPointer)
            cachedTokenPointer = 0
        }

        if (samplerPointer != 0L) {
            llava_sample_free(samplerPointer)
            samplerPointer = 0
        }

        if (embedImagePointer != 0L) {
            try {
                llava_image_embed_free(embedImagePointer)
                embedImagePointer = 0
            } catch (e: Exception) {
                println(e)
            } catch (e: Error) {
                throw RuntimeException("Native method not found: ${e.message}")
            }
        }

        if (llavaCtxPointer != 0L) {
            llava_ctx_free(llavaCtxPointer)
            llavaCtxPointer = 0
        }
    }
}
