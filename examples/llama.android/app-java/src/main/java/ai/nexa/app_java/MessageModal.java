package ai.nexa.app_java;

public class MessageModal {


    private String message;
    private String sender;

    private String imageUri;

    public MessageModal(String message, String sender, String imageUri) {
        this.message = message;
        this.sender = sender;
        this.imageUri = imageUri;
    }


    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public String getSender() {
        return sender;
    }

    public void setSender(String sender) {
        this.sender = sender;
    }

    public String getImageUri() {
        return imageUri;
    }

    public void setImageUri(String imageUri) {
        this.imageUri = imageUri;
    }
}

