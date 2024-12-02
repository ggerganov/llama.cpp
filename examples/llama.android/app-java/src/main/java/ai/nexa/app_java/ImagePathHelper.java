package ai.nexa.app_java;

import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class ImagePathHelper {
    private static final String TAG = "MessageProcessor";
    private final Context context;

    public ImagePathHelper(Context context) {
        this.context = context;
    }

    public String getPathFromUri(String uriString) {
        try {
            Uri uri = Uri.parse(uriString);

            // Handle "content://" scheme
            if ("content".equals(uri.getScheme())) {
                // Handle Google Photos and other document providers
                if (DocumentsContract.isDocumentUri(context, uri)) {
                    final String docId = DocumentsContract.getDocumentId(uri);

                    // MediaStore documents
                    if ("com.android.providers.media.documents".equals(uri.getAuthority())) {
                        final String[] split = docId.split(":");
                        final String type = split[0];
                        Uri contentUri = null;

                        if ("image".equals(type)) {
                            contentUri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI;
                        }

                        final String selection = "_id=?";
                        final String[] selectionArgs = new String[]{split[1]};
                        return getDataColumn(context, contentUri, selection, selectionArgs);
                    }
                }
                // MediaStore (general case)
                return getDataColumn(context, uri, null, null);
            }
            // Handle "file://" scheme
            else if ("file".equals(uri.getScheme())) {
                return uri.getPath();
            }
            // Handle absolute path
            else if (new File(uriString).exists()) {
                return uriString;
            }

            return null;
        } catch (Exception e) {
            Log.e(TAG, "Error getting path from URI: " + uriString, e);
            return null;
        }
    }

    public String copyUriToPrivateFile(Context context, String uriString) throws IOException {
        // 将字符串转换回 Uri
        Uri uri = Uri.parse(uriString);

        // 应用私有目录
        File privateDir = context.getExternalFilesDir("images");
        if (privateDir == null) {
            throw new IOException("Private directory not available");
        }

        // 创建目标文件
        File destFile = new File(privateDir, "temp_image_" + System.currentTimeMillis() + ".jpg");

        try (InputStream inputStream = context.getContentResolver().openInputStream(uri);
             OutputStream outputStream = new FileOutputStream(destFile)) {

            if (inputStream == null) {
                throw new IOException("Failed to open URI input stream");
            }

            // 读取并写入数据
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
        }

        // 返回文件路径
        return destFile.getAbsolutePath();
    }

    private String getDataColumn(Context context, Uri uri, String selection, String[] selectionArgs) {
        final String[] projection = {MediaStore.Images.Media.DATA};
        try (Cursor cursor = context.getContentResolver().query(uri, projection, selection, selectionArgs, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                final int columnIndex = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
                return cursor.getString(columnIndex);
            }
        } catch (Exception e) {
            Log.e(TAG, "Error getting data column", e);
        }
        return null;
    }
}
