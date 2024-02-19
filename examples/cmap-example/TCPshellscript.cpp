// Code to run the terminal shell command `lsof -i :8080' from C++

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

using namespace std;

static string get_port_usage(int port) {
  // Build the command string
  string command = "lsof -i :" + to_string(port);

  // Create a pipe for capturing output
  FILE *pipe = popen(command.c_str(), "r");
  if (!pipe) {
    cerr << "Error opening pipe" << endl;
    return "";
  }

  // Read the output from the pipe
  string output;
  char buffer[128];
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    output += buffer;
  }

  // Close the pipe
  pclose(pipe);

  return output;
}

int main() {
  int port = 8080;
  string output = get_port_usage(port);

  if (output.empty()) {
    cerr << "Error getting port " << port << " usage" << endl;
  } else {
    cout << "Port " << port << " usage:" << endl << output << endl;
  }

  return 0;
}
