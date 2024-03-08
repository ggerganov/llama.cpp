from queue import Queue
import threading
import requests
import json

def print_dict(data):
    if isinstance(data, dict):
        #for k, v in data.items():
        #    print(f"Key: {k}; Value: {v}\n")
        #input("",)
        for k, v in data.items():
            if isinstance(v, dict):
                print_dict(v)
            elif k == "content":
                print(f"Model: {data['model']}")
                print(f"Max tokens predicted: {data['generation_settings']['n_predict']}")
                print(f"Prompt evaluation time = {data['timings']['prompt_ms']}")
                print(f"Token generation time = {data['timings']['predicted_ms']}")
                print(f"Tokens cached = {data['tokens_cached']}")
                print(f"Tokens evaluated = {data['tokens_evaluated']}")
                print(f"Tokens actually predicted = {data['tokens_predicted']}\n")
                print(f"Response: {v}")
                return
    elif isinstance(data, list):
        for entry in v:
            print_dict(entry)
    elif isinstance(data, str):
        print(f"Incoming string is {data}.\n")
    else:
        print("No intelligible data received.\n")
    return

def title_print(text):

    length = len(text)
    print("\n" + "*" * length)
    print(text)
    print("*" * length + "\n")

def make_empty_bar(num_requests):
    bar = []
    for i in range(num_requests):
        bar.append("\u2589")
    bar = ' '.join(bar)
    bar = bar.replace(' ','')
    # print(f"Bar is now {bar}.\n")
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

    system = "You are a helpful assistant who answers all requests \
courteously and accurately without undue repetion. \
You pay close attention to the nuance of a question and respond accordingly."

    data = {'system': system, 'prompt': question}

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code in [200,300]:
            with lockbar:
                #for attr in dir(response.raw):
                    #if not attr.startswith('__'):
                        #print(f"response.raw.{attr} has content {getattr(response.raw, attr)}\n")
                        #input("Press any key ",)
                print(f"Current Client Queue Size: {q.qsize()}; processing request {count} / {num_requests}\n")
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
                    #with lockbar:      # lock automatically releases when the update is done
                    title_print(f"Completed task {count} / {num_requests}")
                    bar = make_progress_bar(bar, count, num_requests)
                q.task_done()
        elif response.status_code == 429 and not q.empty():
            event.set()
            print("Server return too many requests; back off!! Reset event.")
        else:
            print(f"Server responded with code {response.status_code}\n")
    except Exception as e:
        print(f"Server returned exception error {e}")

if __name__ == "__main__":

    global bar
    lockbar = threading.Lock()

    url = "http://192.168.1.28:8080/completion"

    num_requests = 76
    q = Queue(maxsize = 80)
    threads = []

    bar = make_empty_bar(num_requests)

    #api_key = input("What is your API key? ",)
    api_key = "john123456"

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'User-Agent': 'Llamaserver.py',
        'Authorization': f'Bearer {api_key}'
        }

    writer_list = ["Plato", "Aristotle", "Thales of Miletus", "Heraclitus", "Socrates",
                   "The prophet Isaiah", "Jesus of Nazareth", "Plotinus", "Porphyry",
                   "Irenaeus", "Athanasius", "Augustine of Hippo", "Thomas Aquinas", "Anselm of Canterbury",
                   "Roget Bacon", "Fibonacci", "Duns Scotus", "William of Ockham", "Nicholas of Cusa",
                   "Erasmus", "Thomas More", "Luther", "Calvin", "Thomas Cranmer", "Shakespeare",
                   "Francis Bacon", "Thomas Cromwell", "Thomas Hobbes", "John Locke", "David Hume", "Berkeley", "Immanuel Kant",
                   "Jeremy Bentham", "William Blake", "John Stuart Mill", "Peirce", "Ralph Waldo Emerson", "Emily Dickinson", "Walt Whitman", "William James", "Henry James", "Henry Sidgwick", "John Dewey"]

    country_list = ["France", "Germany", "China", "USA", "Italy", "India",
                    "Ukraine", "Japan", "Australia", "New Zealand", "Indonesia", "Nigeria", "Saudi Arabia",
                    "Israel", "Egypt", "Kenya", "Chile", "Mexico", "Canada", "Ecuador", "Brazil", "Argentina", "Colombia",
                    "Bulgaria", "Romania", "Finland", "Sweden", "Norway", "Denmark", "Tanzania", "Israel",
                    "Latvia", "Lithuania", "Estonia", "Pakistan", "Sri Lanka", "Malawi", "Mozambique"]

    philosopher_list = ["Blaise Pascal", "Thomas Hobbes", "Georg Frederik Hegel", "Søren Kierkegaard", "Karl Marx", "Arthur Schopenhauer",
                        "Ludwig Feuerbach", "Friedrich Nietzsche", "Max Weber", "Sigmund Freud", "Carl Jung",
                        "Melanie Klein", "John Puddefoot"]

    num_requests = len(philosopher_list)

    for i in range(num_requests):
        writer = philosopher_list[i % num_requests]
        question = f"Tell me about the writings of {writer}."
        # NOTE: don't pass the parameter as a function call; pass in args
        print(f"Processing request {i} / {num_requests}: {question}\n")
        event = threading.Event()
        t = threading.Thread(target=send_request, args=(q, question, event, i, num_requests))
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()   # wait for all threads to finish

    '''
    print("FINISHED AND GETTING RESULTS")
    while not q.empty():
        text = q.get()
        print_dict(json.loads(text))
    '''
