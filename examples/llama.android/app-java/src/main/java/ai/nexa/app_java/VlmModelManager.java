package ai.nexa.app_java;

import android.content.Context;
import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.io.IOException;

public class VlmModelManager {
    private static final String TAG = "LlamaBridge";
    private static final String MODELS_DIR = "models";
//    private static final String MODEL_TEXT_FILENAME = "nanollava-text-model-q4_0.gguf";
//    private static final String MODEL_MMPROJ_FILENAME = "nanollava-mmproj-f16.gguf";

    private static final String MODEL_TEXT_FILENAME = "model-q8_0.gguf";
    private static final String MODEL_MMPROJ_FILENAME = "projector-q8_0.gguf";

//    private static final String MODEL_TEXT_FILENAME = "nano-vlm-instruct-llm-F16.gguf";
//    private static final String MODEL_MMPROJ_FILENAME = "nano-vlm-instruct-mmproj-F16.gguf";


    private final Context context;
    private File textModelFile;
    private File mmProjModelFile;
    private final File externalModelDir;

    public VlmModelManager(Context context) {
        this.context = context;
        this.externalModelDir = new File(Environment.getExternalStorageDirectory(),
                "Android/data/" + context.getPackageName() + "/files");
    }

    /**
     * Search for model in common locations
     * @param modelFilename The name of the model file to find
     * @return File path to the model if found, null otherwise
     */
    private String findExistingModel(String modelFilename) {
        // List of possible locations to check
        File[] locations = {
                // External storage specific path
                new File(externalModelDir, modelFilename),
                // Downloads folder
                new File(Environment.getExternalStoragePublicDirectory(
                        Environment.DIRECTORY_DOWNLOADS), modelFilename),
                // App's private external storage
                new File(context.getExternalFilesDir(null), MODELS_DIR + "/" + modelFilename),
                // App's private internal storage
                new File(context.getFilesDir(), MODELS_DIR + "/" + modelFilename)
        };

        for (File location : locations) {
            if (location.exists() && location.canRead()) {
                Log.d(TAG, "Found model at: " + location.getAbsolutePath());
                return location.getAbsolutePath();
            }
        }
        return null;
    }

    /**
     * Get text model path, searching in storage locations
     * @return Path to the model file
     * @throws IOException if model cannot be found or accessed
     */
    public String getTextModelPath() throws IOException {
        // If we already have a valid model file, return it
        if (textModelFile != null && textModelFile.exists() && textModelFile.canRead()) {
            return textModelFile.getAbsolutePath();
        }

        // Search for existing model
        String path = findExistingModel(MODEL_TEXT_FILENAME);
        if (path != null) {
            textModelFile = new File(path);
            return path;
        }

        throw new IOException("Text model not found in any storage location");
    }

    /**
     * Get mmproj model path, searching in storage locations
     * @return Path to the model file
     * @throws IOException if model cannot be found or accessed
     */
    public String getMmProjModelPath() throws IOException {
        // If we already have a valid model file, return it
        if (mmProjModelFile != null && mmProjModelFile.exists() && mmProjModelFile.canRead()) {
            return mmProjModelFile.getAbsolutePath();
        }

        // Search for existing model
        String path = findExistingModel(MODEL_MMPROJ_FILENAME);
        if (path != null) {
            mmProjModelFile = new File(path);
            return path;
        }

        throw new IOException("MMProj model not found in any storage location");
    }

    /**
     * Check if both required models exist in any location
     * @return true if both models are found
     */
    public boolean areModelsAvailable() {
        try {
            getTextModelPath();
            getMmProjModelPath();
            return true;
        } catch (IOException e) {
            Log.w(TAG, "Models not available: " + e.getMessage());
            return false;
        }
    }

    /**
     * Get the directory containing the models
     * @return File object for the models directory, or null if models aren't found
     */
    public File getModelsDirectory() {
        try {
            String textModelPath = getTextModelPath();
            return new File(textModelPath).getParentFile();
        } catch (IOException e) {
            Log.w(TAG, "Could not determine models directory: " + e.getMessage());
            return null;
        }
    }
}
