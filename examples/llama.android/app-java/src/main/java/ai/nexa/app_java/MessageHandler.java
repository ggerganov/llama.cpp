package ai.nexa.app_java;

import androidx.recyclerview.widget.RecyclerView;
import android.os.Handler;
import android.os.Looper;

import java.util.ArrayList;

public class MessageHandler {
    private final ArrayList<MessageModal> messageModalArrayList;
    private final MessageRVAdapter messageRVAdapter;
    private final RecyclerView recyclerView;
    private final Handler mainHandler;

    public MessageHandler(ArrayList<MessageModal> messageModalArrayList, MessageRVAdapter messageRVAdapter, RecyclerView recyclerView) {
        this.messageModalArrayList = messageModalArrayList;
        this.messageRVAdapter = messageRVAdapter;
        this.recyclerView = recyclerView;
        this.mainHandler = new Handler(Looper.getMainLooper());
    }

    /**
     * Add a new message to the chat
     */
    public void addMessage(MessageModal message) {
        ensureMainThread(() -> {
            messageModalArrayList.add(message);
            messageRVAdapter.notifyItemInserted(messageModalArrayList.size() - 1);
            scrollToBottom();
        });
    }

    /**
     * Update the last bot message with new token
     */
    public void updateLastBotMessage(String newToken) {
        ensureMainThread(() -> {
            if (!messageModalArrayList.isEmpty()) {
                int lastIndex = messageModalArrayList.size() - 1;
                MessageModal lastMessage = messageModalArrayList.get(lastIndex);

                // If last message is from bot, update it
                if ("bot".equals(lastMessage.getSender())) {
                    String currentMessage = lastMessage.getMessage();
                    lastMessage.setMessage(currentMessage + newToken);
                    messageRVAdapter.notifyItemChanged(lastIndex);
                } else {
                    // Create new bot message
                    MessageModal newMessage = new MessageModal(newToken, "bot", null);
                    messageModalArrayList.add(newMessage);
                    messageRVAdapter.notifyItemInserted(messageModalArrayList.size() - 1);
                }
                scrollToBottom();
            }
        });
    }

    /**
     * Finalize the last bot message with complete response
     */
    public void finalizeLastBotMessage(String completeMessage) {
        ensureMainThread(() -> {
            if (!messageModalArrayList.isEmpty()) {
                int lastIndex = messageModalArrayList.size() - 1;
                MessageModal lastMessage = messageModalArrayList.get(lastIndex);

                if ("bot".equals(lastMessage.getSender())) {
                    lastMessage.setMessage(completeMessage);
                    messageRVAdapter.notifyItemChanged(lastIndex);
                } else {
                    MessageModal newMessage = new MessageModal(completeMessage, "bot", null);
                    messageModalArrayList.add(newMessage);
                    messageRVAdapter.notifyItemInserted(messageModalArrayList.size() - 1);
                }
                scrollToBottom();
            }
        });
    }

    /**
     * Clear all messages from the chat
     */
    public void clearMessages() {
        ensureMainThread(() -> {
            messageModalArrayList.clear();
            messageRVAdapter.notifyDataSetChanged();
        });
    }

    /**
     * Get the last message in the chat
     */
    public MessageModal getLastMessage() {
        if (!messageModalArrayList.isEmpty()) {
            return messageModalArrayList.get(messageModalArrayList.size() - 1);
        }
        return null;
    }

    /**
     * Check if the last message is from the bot
     */
    public boolean isLastMessageFromBot() {
        MessageModal lastMessage = getLastMessage();
        return lastMessage != null && "bot".equals(lastMessage.getSender());
    }

    /**
     * Scroll the RecyclerView to the bottom
     */
    private void scrollToBottom() {
        if (messageModalArrayList.size() > 1) {
            recyclerView.smoothScrollToPosition(messageModalArrayList.size() - 1);
        }
    }

    /**
     * Ensure all UI updates happen on the main thread
     */
    private void ensureMainThread(Runnable action) {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            action.run();
        } else {
            mainHandler.post(action);
        }
    }
}