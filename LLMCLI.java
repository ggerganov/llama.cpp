import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.Scanner;

public class LLMCLI {
    public static void main(String[] args) {
        // Path to the .exe file
        String exePath = "bin/llama-cli.exe";

        System.out.println("Enter -h for help");
        // Scanner to take user input for various commands
        Scanner scanner = new Scanner(System.in);

        while (true) {
            String commandInput = scanner.nextLine();

            // Split user input into command array for ProcessBuilder
            String[] commands = commandInput.split(" ");

            // Create an array to hold both the executable path and the commands
            String[] fullCommand = new String[commands.length + 1];
            fullCommand[0] = exePath;  // First element is the executable path
            System.arraycopy(commands, 0, fullCommand, 1, commands.length);  // Copy the user commands after the exe path

            Process process = null;

            try {
                // Create a ProcessBuilder with the executable and dynamic commands
                ProcessBuilder processBuilder = new ProcessBuilder(fullCommand);

                // Redirect error stream to read both error and output in one stream
                processBuilder.redirectErrorStream(true);

                // Start the process
                process = processBuilder.start();

                // Capture output in a separate thread
                Process finalProcess = process;
                new Thread(() -> {
                    try (BufferedReader reader = new BufferedReader(new InputStreamReader(finalProcess.getInputStream()))) {
                        String line;
                        while ((line = reader.readLine()) != null) {
                            System.out.println(line);
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }).start();

                // Use OutputStream to send input to the process (if needed)
                try (OutputStream processInput = process.getOutputStream()) {
                    String userInput;
                    while (scanner.hasNextLine() && process.isAlive()) {
                        userInput = scanner.nextLine();
                        processInput.write((userInput + "\n").getBytes());
                        processInput.flush();  // Ensure input is sent immediately
                    }
                }

                // Wait for the process to complete and get the exit code
                int exitCode = process.waitFor();
            } catch (IOException | InterruptedException e) {
                e.printStackTrace();
            } finally {
                // Ensure the process is destroyed if still running
                if (process != null) {
                    process.destroy();
                }
            }
        }
    }
}