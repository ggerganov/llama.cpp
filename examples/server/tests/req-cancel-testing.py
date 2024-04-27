import threading
import requests

# Stats
total_requests = 0
requests_executed = 0
requests_cancelled = 0
requests_remaining = 0

class StoppableThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def stopped(self):
        return self.stop_event.is_set()

def send_request(stop_event):
    try:
        url = 'http://127.0.0.1:8080/completion'
        data = {
            'prompt': 'Hello llama',
            'n_predict': 2
        }
        if not stop_event.is_set():
            response = requests.post(url, json=data, timeout=60)  # Reduced timeout for testing
            print('Response:', response.text)
            global requests_executed
            requests_executed += 1
    except requests.exceptions.Timeout:
        print('Request timed out')
    except Exception as e:
        print('An error occurred:', str(e))

def get_health():
    try:
        url = 'http://127.0.0.1:8080/health'
        response = requests.get(url, timeout=10)
        return response.status_code
    except requests.exceptions.Timeout:
        print('Health check timed out')
        return
    except Exception as e:
        print('An error occurred during health check:', str(e))
        return


# User input for the number of requests
num_requests = int(input("How many requests would you like to post?\n"))

total_requests = num_requests

# Launching multiple requests
for i in range(num_requests):
    health = get_health()
    ok_status = False ##our server status

    if health == 503 or health == 500 or health == 200:
        ok_status = True
    
    if ok_status == False:
        print(f"Server is not running. Status:{health}. Exiting now...\n")
        requests_cancelled = total_requests - i
        break
    
    stop_event = threading.Event()
    req_thread = StoppableThread(target=send_request, args=(stop_event,))
    req_thread.start()

    input("Press Enter when request is complete or you would like to stop the request!\n")
    if not stop_event.is_set():
        stop_event.set()

    req_thread.join()  # Ensure the thread finishes

requests_remaining = total_requests - requests_executed - requests_cancelled

print("\nSummary:")
print(f"Total requests: {total_requests}")
print(f"Requests executed: {requests_executed}")
print(f"Requests cancelled: {requests_cancelled}")
print(f"Requests remaining: {requests_remaining}")


