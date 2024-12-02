package ai.nexa.app_java;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.os.Message;
import android.provider.MediaStore;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "ChatApp";
    private static final int PICK_IMAGE_REQUEST = 30311;
    private static final int REQUEST_RECORD_AUDIO_PERMISSION = 200;
    private static final int READ_EXTERNAL_STORAGE_PERMISSION = 303;

    private RecyclerView chatsRV;
    private ImageButton selectImageButton;
    private ImageButton sendMsgIB;
    private EditText userMsgEdt;
    private String justSelectedImageUri;

    private LinearLayout linearLayout;
    private TextView titleAfterChatTextView;
    private RecyclerView recyclerView;

    private ArrayList<MessageModal> messageModalArrayList;
    private MessageRVAdapter messageRVAdapter;
    private MessageHandler messageHandler;
    private LlamaBridge llamaBridge;
    private SpeechRecognizer speechRecognizer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Log.d(TAG, "onCreate: Starting MainActivity");

        initializeViews();
        setupRecyclerView();
        initializeLlamaBridge();
        createSpeechRecognizerIntent();
        setupClickListeners();

        Log.d(TAG, "onCreate: MainActivity setup complete");
    }

    private void initializeViews() {
        chatsRV = findViewById(R.id.idRVChats);
        selectImageButton = findViewById(R.id.btnUploadImage);
        sendMsgIB = findViewById(R.id.idIBSend);
        userMsgEdt = findViewById(R.id.idEdtMessage);
        linearLayout = findViewById(R.id.idLayoutBeforeChat);
        titleAfterChatTextView = findViewById(R.id.textView);
        recyclerView = findViewById(R.id.idRVChats);
    }

    private void setupRecyclerView() {
        messageModalArrayList = new ArrayList<>();
        messageRVAdapter = new MessageRVAdapter(messageModalArrayList, this);
        chatsRV.setLayoutManager(new LinearLayoutManager(this, RecyclerView.VERTICAL, false));
        chatsRV.setAdapter(messageRVAdapter);
        messageHandler = new MessageHandler(messageModalArrayList, messageRVAdapter, recyclerView);
    }

    private void initializeLlamaBridge() {
        llamaBridge = new LlamaBridge(this, messageHandler);
        if (!llamaBridge.areModelsAvailable()) {
            Toast.makeText(this, "Required model files are not available", Toast.LENGTH_LONG).show();
            return;
        }
        llamaBridge.loadModel();
    }

    private void setupClickListeners() {
        selectImageButton.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(intent, PICK_IMAGE_REQUEST);
        });

        sendMsgIB.setOnClickListener(v -> {
            hideKeyboard(v);
            sendTextMessage();
        });
    }

    private void updateChatBotDisplay() {
        linearLayout.setVisibility(View.GONE);
        titleAfterChatTextView.setVisibility(View.VISIBLE);
        recyclerView.setVisibility(View.VISIBLE);
    }

    private void sendTextMessage() {
        updateChatBotDisplay();

        String userMessage = userMsgEdt.getText().toString().trim();
        if (!userMessage.isEmpty()) {
            Log.d(TAG, "Sending message: " + userMessage);
            messageHandler.addMessage(new MessageModal(userMessage, "user", null));

            if (justSelectedImageUri == null) {
                messageHandler.addMessage(new MessageModal("Please select an image first.", "bot", null));
                return;
            }

            // Use LlamaBridge for inference
            llamaBridge.processMessage(userMessage, justSelectedImageUri, new LlamaBridge.InferenceCallback() {
                @Override
                public void onStart() {
                    // Optional: Show loading indicator
                }

                @Override
                public void onToken(String token) {
                    // Update the UI with each token as it comes in
                    runOnUiThread(() -> {
                        messageHandler.updateLastBotMessage(token);
                    });
                }

                @Override
                public void onComplete(String fullResponse) {
                    // Final update with complete response
                    runOnUiThread(() -> {
                        messageHandler.finalizeLastBotMessage(fullResponse);
                    });
                }

                @Override
                public void onError(String error) {
                    runOnUiThread(() -> {
                        Toast.makeText(MainActivity.this, "Error: " + error, Toast.LENGTH_SHORT).show();
                        messageHandler.addMessage(new MessageModal("Error processing message: " + error, "assistant", null));
                    });
                }
            });

            userMsgEdt.setText(""); // Clear the input field after sending
            justSelectedImageUri = null; // Clear the image URI after sending
        } else {
            Toast.makeText(MainActivity.this, "Please enter your message.", Toast.LENGTH_SHORT).show();
        }
    }

    private void sendImageAsMessage(String imageUri) {
        updateChatBotDisplay();
        messageHandler.addMessage(new MessageModal("", "user", imageUri));
        justSelectedImageUri = imageUri;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (llamaBridge != null) {
            llamaBridge.shutdown();
        }
        if (speechRecognizer != null) {
            speechRecognizer.destroy();
        }
    }

    private void createSpeechRecognizerIntent() {
        requestMicrophonePermission();

        ImageButton btnStart = findViewById(R.id.btnStart);

        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this);

        Intent speechRecognizerIntent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        speechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        speechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault());
        speechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true);

        speechRecognizer.setRecognitionListener(new android.speech.RecognitionListener() {
            @Override
            public void onReadyForSpeech(Bundle params) {
            }

            @Override
            public void onBeginningOfSpeech() {
            }

            @Override
            public void onRmsChanged(float rmsdB) {
            }

            @Override
            public void onBufferReceived(byte[] buffer) {
            }

            @Override
            public void onEndOfSpeech() {
            }

            @Override
            public void onError(int error) {
                String errorMessage = getErrorText(error);
                Log.d("SpeechRecognition", "Error occurred: " + errorMessage);
            }

            public String getErrorText(int errorCode) {
                String message;
                switch (errorCode) {
                    case SpeechRecognizer.ERROR_AUDIO:
                        message = "Audio recording error";
                        break;
                    case SpeechRecognizer.ERROR_CLIENT:
                        message = "Client side error";
                        break;
                    case SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS:
                        message = "Insufficient permissions";
                        break;
                    case SpeechRecognizer.ERROR_NETWORK:
                        message = "Network error";
                        break;
                    case SpeechRecognizer.ERROR_NETWORK_TIMEOUT:
                        message = "Network timeout";
                        break;
                    case SpeechRecognizer.ERROR_NO_MATCH:
                        message = "No match";
                        break;
                    case SpeechRecognizer.ERROR_RECOGNIZER_BUSY:
                        message = "RecognitionService busy";
                        break;
                    case SpeechRecognizer.ERROR_SERVER:
                        message = "Error from server";
                        break;
                    case SpeechRecognizer.ERROR_SPEECH_TIMEOUT:
                        message = "No speech input";
                        break;
                    default:
                        message = "Didn't understand, please try again.";
                        break;
                }
                return message;
            }

            @Override
            public void onResults(Bundle results) {
                ArrayList<String> matches = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                if (matches != null && !matches.isEmpty()) {
                    userMsgEdt.setText(matches.get(0)); // Set the recognized text to the EditText
                    sendTextMessage();
                }
            }

            @Override
            public void onPartialResults(Bundle partialResults) {
                // This is called for partial results
                ArrayList<String> partialMatches = partialResults.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                if (partialMatches != null && !partialMatches.isEmpty()) {
                    userMsgEdt.setText(partialMatches.get(0)); // Update EditText with the partial result
                }
            }

            @Override
            public void onEvent(int eventType, Bundle params) {
            }
        });

        btnStart.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                switch (event.getAction()) {
                    case MotionEvent.ACTION_DOWN:
                        // Button is pressed
                        speechRecognizer.startListening(speechRecognizerIntent);
                        return true; // Return true to indicate the event was handled
                    case MotionEvent.ACTION_UP:
                        // Button is released
                        speechRecognizer.stopListening();
                        return true; // Return true to indicate the event was handled
                }
                return false; // Return false for other actions
            }
        });
    }

    private void requestMicrophonePermission() {
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO_PERMISSION);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case READ_EXTERNAL_STORAGE_PERMISSION:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(this, "Read External Storage Permission Granted", Toast.LENGTH_SHORT).show();
                    Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                    startActivityForResult(intent, PICK_IMAGE_REQUEST);
                } else {
                    Toast.makeText(this, "Read External Storage Permission Denied", Toast.LENGTH_SHORT).show();
                }
                break;
            default:
                break;
        }

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null) {
            Uri selectedImage = data.getData();
            if (selectedImage != null) {
                String imageUriString = selectedImage.toString();
                sendImageAsMessage(imageUriString);
            }
        }
    }

    public void hideKeyboard(View view) {
        InputMethodManager inputMethodManager = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
        if (inputMethodManager != null) {
            inputMethodManager.hideSoftInputFromWindow(view.getWindowToken(), InputMethodManager.HIDE_NOT_ALWAYS);
        }
    }

}