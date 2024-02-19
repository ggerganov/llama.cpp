from queue import Queue
import threading
import requests
import json
from time import sleep

def print_dict(data):
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                print_dict(v)
            elif k == "content":          
                print(f"Key: {k:>30}: {v}")
                return
    elif isinstance(data, list):
        for entry in v:
            print_dict(entry)
    elif isinstance(data, str):
        print(f"Incoming string is {data}.\n")
    return

def print_response(text):
    print(text)

def make_empty_bar(num_requests):
    bar = []
    for i in range(num_requests):
        bar.append("\u2589")
    bar = ' '.join(bar)
    bar = bar.replace(' ','')
    print(f"Bar is now {bar}.\n")
    return bar

def make_progress_bar(bar, count, num_requests):
    stride1 = len("\u2589")
    stride2 = len("\u23F1")
    for i in range(num_requests):
        if i == count:
            # print(f"Bar position {i} is {bar[i]}\n")
            bar = bar[:i*stride1] + "\u23F1" + bar[i*stride1 + stride2:]
    print(f"Bar is now {bar}\n")
    return bar

def send_request(q, question, event, count, num_requests):

    delay = 0.1

    global bar

    data = {'prompt': question}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code in [200,300]:
            print(f"Current Queue Size: {q.qsize()}; processing request {count} / {num_requests}\n")
            print(f"Status Code for {question}: {response.status_code}\n")
            print(f"Response to {question}:\n")
            if isinstance(response.text, str):
                data = json.loads(response.text)
                if isinstance(data, dict):
                    print_dict(data)
                elif isinstance(data, str):
                    print(data)
                else:
                    print("\nServer returned data of wrong type.\n")
            # put the response text in the queue
            q.put(response.text)
            if not q.empty():
                print(f"Completed task {count} / {num_requests}\n")
                bar = make_progress_bar(bar, count, num_requests)
            q.task_done()
        elif response.status_code == 429 and not q.empty():
            event.set()
            print("Server return too many requests; back off!! Reset event.")
        else:
            print(f"Server responded with code {response.status_code}\n")
    except Exception as e:
        print(f"Server returned exception error {e}")
        sleep(delay)
        delay *= 2

if __name__ == "__main__":

    global bar
    
    url = "http://localhost:8080/completion"

    num_requests = 40
    q = Queue(maxsize = 40)
    threads = []

    bar = make_empty_bar(num_requests)

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',  
        'User-Agent': 'Llamaserver.py'
        }

    country_list = ["France", "Germany", "China", "USA", "Italy", "India",
                    "Ukraine", "Japan", "Australia", "New Zealand", "Indonesia", "Nigeria", "Saudi Arabia",
                    "Israel", "Egypt", "Kenya", "Chile", "Mexico", "Canada",
                    "Bulgaria", "Romania", "Finland", "Sweden", "Norway", "Denmark", "Tanzania", "Israel",
                    "Latvia", "Lithuania", "Estonia", "Pakistan", "Sri Lanka", "Malawi", "Mozambique"]
    
    for i in range(num_requests):
        country = country_list[i % len(country_list)]
        question = f"When was the first democratic election (if any) in {country}?"
        # NOTE: don't pass the parameter as a function call; pass in args
        print(f"Processing request {i} / {num_requests}: {question}\n")
        event = threading.Event()
        t = threading.Thread(target=send_request, args=(q, question, event, i, num_requests)) 
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()   # wait for all threads to finish

    print("FINISHED AND GETTING RESULTS")
    while not q.empty():
        text = q.get()  
        print_dict(json.loads(text))


    


