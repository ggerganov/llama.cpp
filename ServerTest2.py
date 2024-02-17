import threading
import queue
import requests

def print_dict(data):
    if isinstance(data, dict):
        print_dict(data)
    elif isinstance(data, list):
        for entry in data:
            print_dict(entry)
    elif data == "content":          
        print(f"Key: {data:>30}: {data['content']}")
    return

def producer(list):
    # Generate test requests and add them to the queue
    for i in range(10):  # Adjust for desired load
        request_data = f"What is the capital of {list[i % len(list)]}?"
        print(f"Request: {request_data}")
        requests_queue.put(request_data)

def consumer():
    while True:
        try:
            request_data = requests_queue.get()
            print(f"Processing {request_data}")
            response = requests.post("http://localhost:8080", data=request_data)
            print_dict(response.text)
        except Exception as e:
            print(f"Exception {e}\n")
            continue
        finally:
            requests_queue.task_done()

# Define your test request data
requests_queue = queue.Queue()

# number of threads
num_threads = 5

# some text data
country_list = ["France", "Germany", "China", "USA", "Italy", "India",
    "Ukraine", "Japan", "Australia", "New Zealand", "Indonesia", "Nigeria", "Saudi Arabia", "Israel", "Egypt", "Kenya", "Chile", "Mexico", "Canada"]

# Create producer and consumer threads
producer_thread = threading.Thread(target=producer, args = (country_list,))
consumer_threads = [threading.Thread(target=consumer) for _ in range(num_threads)]  # Adjust thread count

# Start threads and monitor resources
producer_thread.start()
for thread in consumer_threads:
    thread.start()

producer_thread.join()
for thread in consumer_threads:
    thread.join()

print("Stress test completed!")
