# running Mixtral in a loop

# Needs a zsh change to max memory using
# sudo sysctl iogpu.wired_limit_mb=27500 (anything bigger crashes easily)

import os
import subprocess
import re
import psutil
import threading
import time
import queue

def get_pid():
    # Get the parent process ID (PPID) of the current Python script
    current_pid = os.getpid()
    parent_pid = None

    # Iterate through all the parent processes to find the actual Python process
    while parent_pid is not None:
        try:
            parent_proc = psutil.Process(parent_pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            parent_pid = None
        else:
            if 'python' in parent_proc.name():
                current_pid = parent_pid
            else:
                parent_pid = parent_proc.ppid()

    # Print the PID of the running Python script
    print(f"The PID of the running Python script is: {current_pid}")

    return current_pid

def get_cpu_percent():
    cpu_percent = psutil.cpu_percent()  # Measure CPU usage every second
    return cpu_percent

def get_memory_info():
    mem_info = psutil.virtual_memory()
    return {
        'total': mem_info.total,
        'used': mem_info.used,
        'percent': mem_info.percent
    }

def get_threads():
    # Get the PID of the process you want to inspect
    pid = get_pid()

    # Get the process object
    process = psutil.Process(pid)

    # Print the number of threads used by the process
    print("Number of threads:", len(process.threads()))

    # Iterate over the threads and print their attributes
    for thread in process.threads():
        print(f"Thread ID: {thread.id}")
        #print(f"Thread count: {thread.count}")
        #print(f"Thread index: {thread.index}")
        print(f"Thread system_time: {thread.system_time}")
        print(f"Thread user time: {thread.user_time}")

def find_time_and_tokens(string):
    # Define the regular expression pattern
    pattern = r"llama_print_timings:       total time =\s*(\d+(\.\d+)?)\s*ms /\s*(\d+)"
    pattern2 = r"llama_model_loader: - kv  10:                    llama.expert_used_count u32              = (\d+)"

    # Search for the pattern in stderr
    match = re.search(pattern, string)
    match2 = re.search(pattern2, string)

    if match:
        # Extract the total time and token count from the matched groups
        total_time = float(match.group(1))
        token_count = int(match.group(3))

        print(f"Total time taken: {total_time} ms")
        print(f"Token consumption count: {token_count}")
    else:
        print("Could not find the total time and token count in the output.")

    if match2:
        # Extract the total time and token count from the matched groups
        experts_used = float(match2.group(1))

        print(f"Number of experts used: {experts_used}")
    else:
        print("Could not find the total number of experts used in the process.")

def command_setup(return_queue, prompt="How can I use python psutil package to calculate CPU and memory usage in a run?"):

    prompt2 = f" [INST] {prompt} [/INST] "
    kv_override = f"llama_kv_expert_used_count=int:3"
    command = [
        '/Users/edsilm2/llama.cpp/build/bin/main',
        '-m',
        '/Users/edsilm2/llama.cpp/models/Mixtral-8x7b-Q2_K/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf',
        '-p', prompt2,
        '-ngl', '99',
        '-c', '4096',
        '-n', '-1',
        '-s', '1',
        '-ctk', 'q8_0',
        '--override-kv', kv_override    # this doesn't have any effect on the LOG which doesn't reflect kv overrides (they say)
        ]


    #print(command)
    response = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    exit_code = response.wait()
    # print(dir(response))
    # print("Returned from subprocess call.")
    stdout, stderr = response.communicate()

            # Check if the command was successful (exit code 0 usually means success)
    if exit_code == 0:
        print(f"\nUser input: \033[31m{prompt}\033[0m\n")
        # Convert the output bytes to a string and print it
        output_str = stdout.decode('utf-8').strip()
        print(f"Output: \033[33m{output_str}\033[0m\n")

        output_err = stderr.decode('utf-8').strip()
        #print(f"Standard Error: \033[33m{output_err}\033[0m\n")
    else:
        try:
            # There was an error, print the error message
            error_str = stderr.decode('utf-8').strip()
            print('Error:', error_str)
        except AttributeError as ae:
            print(f"Unable to process the exit code correctly: {ae}.")

    find_time_and_tokens(output_err)

    cpu_percent_usage = get_cpu_percent()
    print(f"CPU percentage usage = {cpu_percent_usage}\n")

    get_threads()

    memory_info = get_memory_info()
    print(f"Memory usage: Total = {memory_info['total']} Used = {memory_info['used']} Percentage = {memory_info['percent']}")

    # Put return values on queue
    return_queue.put((stdout, stderr, exit_code))

def check_response(response):
    start = time.time()
    while time.time() - start < 30:
        if response.poll() is not None:
            break
        time.sleep(1)

    if response.poll() is None:
        print("Killing process")
        response.kill()

if __name__ == "__main__":

    prompt = "Who are you?"
    while prompt != "quit":

        # original user prompt was here

        q = queue.Queue()

        #response, error, code = command_setup(prompt)

        thread = threading.Thread(target=command_setup, args=(q, prompt))
        thread.start()

        # Wait with timeout
        thread.join(timeout=5)

        # Get return values from queue
        if not q.empty():
            stdout, stderr, exit_code = q.get()

        prompt = input("Awaiting the reply from mixtral ... ",)
