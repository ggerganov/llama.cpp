package ai.nexa.app_java;

import android.content.Context;
import com.nexa.NexaOmniVlmInference;
import android.util.Log;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import kotlin.Unit;
import kotlin.coroutines.Continuation;
import kotlin.jvm.functions.Function1;
import kotlinx.coroutines.BuildersKt;
import kotlinx.coroutines.CoroutineStart;
import kotlinx.coroutines.Dispatchers;
import kotlinx.coroutines.GlobalScope;
import kotlinx.coroutines.Job;
import kotlinx.coroutines.flow.Flow;
import kotlinx.coroutines.flow.FlowCollector;

public class LlamaBridge {
    private static final String TAG = "LlamaBridge";
    private final Context context;
    private final ExecutorService executor;
    private final MessageHandler messageHandler;
    private final VlmModelManager modelManager;
    private final ImagePathHelper imagePathHelper;
    private NexaOmniVlmInference nexaVlmInference;
    private boolean isModelLoaded = false;

    private final KotlinFlowHelper flowHelper = new KotlinFlowHelper();

    // Default inference parameters
    private static final float DEFAULT_TEMPERATURE = 1.0f;
    private static final int DEFAULT_MAX_TOKENS = 64;
    private static final int DEFAULT_TOP_K = 50;
    private static final float DEFAULT_TOP_P = 0.9f;

    public interface InferenceCallback {
        void onStart();
        void onToken(String token);
        void onComplete(String fullResponse);
        void onError(String error);
    }

    public LlamaBridge(Context context, MessageHandler messageHandler) {
        this.context = context;
        this.messageHandler = messageHandler;
        this.executor = Executors.newSingleThreadExecutor();
        this.modelManager = new VlmModelManager(context);
        this.imagePathHelper = new ImagePathHelper(context);
    }

    public boolean areModelsAvailable() {
        return modelManager.areModelsAvailable();
    }

    public void loadModel() {
        executor.execute(() -> {
            try {
                if (!modelManager.areModelsAvailable()) {
                    throw new IOException("Required model files are not available");
                }

                String modelPath = modelManager.getTextModelPath();
                String projectorPath = modelManager.getMmProjModelPath();

                Log.d(TAG, "Loading model from: " + modelPath);
                Log.d(TAG, "Loading projector from: " + projectorPath);

                // Create with default values for optional parameters
                nexaVlmInference = new NexaOmniVlmInference(
                        modelPath,            // modelPath
                        projectorPath,        // projectorPath
                        "",                   // imagePath (empty string as default)
                        new ArrayList<>(Arrays.asList("</s>")),    // stopWords (empty list)
                        DEFAULT_TEMPERATURE,  // temperature
                        DEFAULT_MAX_TOKENS,   // maxNewTokens
                        DEFAULT_TOP_K,        // topK
                        DEFAULT_TOP_P         // topP
                );
                nexaVlmInference.loadModel();
                isModelLoaded = true;

                Log.d(TAG, "Model loaded successfully.");
//                messageHandler.addMessage(new MessageModal("Model loaded successfully", "assistant", null));
            } catch (Exception e) {
                Log.e(TAG, "Failed to load model", e);
                messageHandler.addMessage(new MessageModal("Error loading model: " + e.getMessage(), "assistant", null));
            }
        });
    }

//    public void processMessage(String message, String imageUri, InferenceCallback callback) {
//        if (!isModelLoaded) {
//            callback.onError("Model not loaded yet");
//            return;
//        }
//
//        try {
//            // Add user message first
//            MessageModal userMessage = new MessageModal(message, "user", imageUri);
//            messageHandler.addMessage(userMessage);
//
//            // Create an initial empty assistant message
//            MessageModal assistantMessage = new MessageModal("", "assistant", null);
//            messageHandler.addMessage(assistantMessage);
//
//            // Convert image URI to absolute path
//            String imageAbsolutePath = imagePathHelper.getPathFromUri(imageUri);
//
//            Flow<String> flow = nexaVlmInference.createCompletionStream(
//                    message,
//                    imageAbsolutePath,
//                    new ArrayList<>(),
//                    DEFAULT_TEMPERATURE,
//                    DEFAULT_MAX_TOKENS,
//                    DEFAULT_TOP_K,
//                    DEFAULT_TOP_P
//            );
//
//            if (flow != null) {
//                CoroutineScope scope = CoroutineScopeKt.CoroutineScope(Dispatchers.getMain());
//
//                Job job = FlowKt.launchIn(
//                        FlowKt.onEach(flow, new Function2<String, Continuation<? super Unit>, Object>() {
//                            @Override
//                            public Object invoke(String token, Continuation<? super Unit> continuation) {
//                                messageHandler.updateLastAssistantMessage(token);
//                                callback.onToken(token);
//                                return Unit.INSTANCE;
//                            }
//                        }),
//                        scope
//                );
//            } else {
//                messageHandler.finalizeLastAssistantMessage("Error: Failed to create completion stream");
//                callback.onError("Failed to create completion stream");
//            }
//        } catch (Exception e) {
//            Log.e(TAG, "Error processing message", e);
//            messageHandler.finalizeLastAssistantMessage("Error: " + e.getMessage());
//            callback.onError(e.getMessage());
//        }
//    }

    public void processMessage(String message, String imageUri, InferenceCallback callback) {
        if (!isModelLoaded) {
            callback.onError("Model not loaded yet");
            return;
        }

        String imageAbsolutePath = null;
        try {
            imageAbsolutePath = imagePathHelper.copyUriToPrivateFile(context, imageUri);
        } catch (IOException e) {
            callback.onError("Failed to process image: " + e.getMessage());
            return;
        }

        final String imagePath = imageAbsolutePath;
        MessageModal assistantMessage = new MessageModal("", "bot", null);
        messageHandler.addMessage(assistantMessage);

        try {
            Flow<String> flow = nexaVlmInference.createCompletionStream(
                    message,
                    imagePath,
                    new ArrayList<>(Arrays.asList("</s>")),
                    DEFAULT_TEMPERATURE,
                    DEFAULT_MAX_TOKENS,
                    DEFAULT_TOP_K,
                    DEFAULT_TOP_P
            );

            callback.onStart();
            StringBuilder fullResponse = new StringBuilder();

            Job collectJob = BuildersKt.launch(
                    GlobalScope.INSTANCE,
                    Dispatchers.getIO(),
                    CoroutineStart.DEFAULT,
                    (coroutineScope, continuation) -> {
                        flow.collect(new FlowCollector<String>() {
                            @Override
                            public Object emit(String token, Continuation<? super Unit> continuation) {
                                fullResponse.append(token);
                                callback.onToken(token);
                                return Unit.INSTANCE;
                            }
                        }, continuation);
                        callback.onComplete(fullResponse.toString());
                        return Unit.INSTANCE;
                    }
            );

            collectJob.invokeOnCompletion(new Function1<Throwable, Unit>() {
                @Override
                public Unit invoke(Throwable throwable) {
                    if (throwable != null && !(throwable instanceof CancellationException)) {
                        callback.onError("Stream collection failed: " + throwable.getMessage());
                    }
                    return Unit.INSTANCE;
                }
            });

        } catch (Exception e) {
            Log.e(TAG, "Inference failed", e);
            callback.onError(e.getMessage());
        }
    }

    public void cleanup() {
        flowHelper.cancel();
    }

//    public void processMessageWithParams(
//            String message,
//            String imageUri,
//            float temperature,
//            int maxTokens,
//            int topK,
//            float topP,
//            InferenceCallback callback) {
//
//        if (!isModelLoaded) {
//            callback.onError("Model not loaded yet");
//            return;
//        }
//
//        executor.execute(() -> {
//            StringBuilder fullResponse = new StringBuilder();
//            try {
//                callback.onStart();
//
//                Flow<String> completionStream = nexaVlmInference.createCompletionStream(
//                        message,
//                        imageUri,
//                        new ArrayList<>(),
//                        temperature,
//                        maxTokens,
//                        topK,
//                        topP
//                );
//
//                completionStream.collect(new FlowCollector<String>() {
//                    @Override
//                    public Object emit(String value, Continuation<? super Unit> continuation) {
//                        fullResponse.append(value);
//                        callback.onToken(value);
//                        return Unit.INSTANCE;
//                    }
//                });
//
//                callback.onComplete(fullResponse.toString());
//
//            } catch (Exception e) {
//                Log.e(TAG, "Inference failed", e);
//                callback.onError(e.getMessage());
//            }
//        });
//    }


    public void shutdown() {
        if (nexaVlmInference != null) {
            executor.execute(() -> {
                try {
                    nexaVlmInference.dispose();
                } catch (Exception e) {
                    Log.e(TAG, "Error closing inference", e);
                }
                nexaVlmInference = null;
                isModelLoaded = false;
            });
        }
        executor.shutdown();
    }
}