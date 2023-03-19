# A hacky little script from Concedo that exposes llama.cpp function bindings 
# allowing it to be used via a simulated kobold api endpoint
# it's not very usable as there is a fundamental flaw with llama.cpp 
# which causes generation delay to scale linearly with original prompt length.

import ctypes
import os

class load_model_inputs(ctypes.Structure):
    _fields_ = [("threads", ctypes.c_int),
                ("max_context_length", ctypes.c_int),
                ("batch_size", ctypes.c_int),
                ("model_filename", ctypes.c_char_p),
                ("n_parts_overwrite", ctypes.c_int)]

class generation_inputs(ctypes.Structure):
    _fields_ = [("seed", ctypes.c_int),
                ("prompt", ctypes.c_char_p),
                ("max_length", ctypes.c_int),
                ("temperature", ctypes.c_float),
                ("top_k", ctypes.c_int),
                ("top_p", ctypes.c_float),
                ("rep_pen", ctypes.c_float),
                ("rep_pen_range", ctypes.c_int)]

class generation_outputs(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int),
                ("text", ctypes.c_char * 16384)]

dir_path = os.path.dirname(os.path.realpath(__file__))
handle = ctypes.CDLL(dir_path + "/llamacpp.dll")     

handle.load_model.argtypes = [load_model_inputs] 
handle.load_model.restype = ctypes.c_bool
handle.generate.argtypes = [generation_inputs]
handle.generate.restype = generation_outputs
  
def load_model(model_filename,batch_size=8,max_context_length=512,threads=4,n_parts_overwrite=-1):
    inputs = load_model_inputs()
    inputs.model_filename = model_filename.encode("UTF-8")
    inputs.batch_size = batch_size
    inputs.max_context_length = max_context_length
    inputs.threads = threads
    inputs.n_parts_overwrite = n_parts_overwrite
    ret = handle.load_model(inputs)
    return ret

def generate(prompt,max_length=20,temperature=0.8,top_k=100,top_p=0.85,rep_pen=1.1,rep_pen_range=128,seed=-1):
    inputs = generation_inputs()
    outputs = generation_outputs()
    inputs.prompt = prompt.encode("UTF-8")
    inputs.max_length = max_length
    inputs.temperature = temperature
    inputs.top_k = top_k
    inputs.top_p = top_p
    inputs.rep_pen = rep_pen
    inputs.rep_pen_range = rep_pen_range
    inputs.seed = seed
    ret = handle.generate(inputs,outputs)
    if(ret.status==1):
        return ret.text.decode("UTF-8")
    return ""


#################################################################
### A hacky simple HTTP server simulating a kobold api by Concedo
### we are intentionally NOT using flask, because we want MINIMAL dependencies
#################################################################
import json, http.server, threading, socket, sys, time

# global vars
global modelname 
modelname = ""
maxctx = 1024
maxlen = 256
modelbusy = False
port = 5001

class ServerRequestHandler(http.server.BaseHTTPRequestHandler):

    sys_version = ""
    server_version = "ConcedoLlamaForKoboldServer"

    def do_GET(self):
        if not self.path.endswith('/'):
            # redirect browser
            self.send_response(301)
            self.send_header("Location", self.path + "/")
            self.end_headers()
            return

        if self.path.endswith('/api/v1/model/') or self.path.endswith('/api/latest/model/'):
            self.send_response(200)
            self.end_headers()
            global modelname
            self.wfile.write(json.dumps({"result": modelname }).encode())
            return

        if self.path.endswith('/api/v1/config/max_length/') or self.path.endswith('/api/latest/config/max_length/'):
            self.send_response(200)
            self.end_headers()
            global maxlen
            self.wfile.write(json.dumps({"value":maxlen}).encode())
            return

        if self.path.endswith('/api/v1/config/max_context_length/') or self.path.endswith('/api/latest/config/max_context_length/'):
            self.send_response(200)
            self.end_headers()
            global maxctx
            self.wfile.write(json.dumps({"value":maxctx}).encode())
            return
        
        self.send_response(404)
        self.end_headers()
        rp = 'Error: HTTP Server is running, but this endpoint does not exist. Please check the URL.'
        self.wfile.write(rp.encode())
        return

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        if self.path.endswith('/api/v1/generate/') or self.path.endswith('/api/latest/generate/'):
            global modelbusy  
            if modelbusy:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(json.dumps({"detail": {
                        "msg": "Server is busy; please try again later.",
                        "type": "service_unavailable",
                    }}).encode())
                return    
            else:      
                modelbusy = True
                genparams = None
                try:
                    genparams = json.loads(body)
                except ValueError as e:
                    self.send_response(503)
                    self.end_headers()
                    return                
                 
                print("\nInput: " + json.dumps(genparams))
                recvtxt = generate(
                    prompt=genparams.get('prompt', ""),
                    max_length=genparams.get('max_length', 50),
                    temperature=genparams.get('temperature', 0.8),
                    top_k=genparams.get('top_k', 100),
                    top_p=genparams.get('top_p', 0.85),
                    rep_pen=genparams.get('rep_pen', 1.1),
                    rep_pen_range=genparams.get('rep_pen_range', 128),
                    seed=-1
                    )
                print("\nOutput: " + recvtxt)
                res = {"results": [{"text": recvtxt}]}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(res).encode())
                modelbusy = False
                return
        self.send_response(404)
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()
    
    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Content-type', 'application/json')
        return super(ServerRequestHandler, self).end_headers()


def RunServerMultiThreaded(port, HandlerClass = ServerRequestHandler,
         ServerClass = http.server.HTTPServer):
    addr = ('', port)
    sock = socket.socket (socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(addr)
    sock.listen(5)

    # Start listener threads.
    class Thread(threading.Thread):
        def __init__(self, i):
            threading.Thread.__init__(self)
            self.i = i
            self.daemon = True
            self.start()
        def run(self):
            with http.server.HTTPServer(addr, HandlerClass, False) as self.httpd:
                #print("Thread %s - Web Server is running at http://0.0.0.0:%s" % (self.i, port))
                try:
                    self.httpd.socket = sock
                    self.httpd.server_bind = self.server_close = lambda self: None
                    self.httpd.serve_forever()
                except (KeyboardInterrupt,SystemExit):
                    #print("Thread %s - Server Closing" % (self.i))
                    self.httpd.server_close()
                    sys.exit(0)
                finally:
                    # Clean-up server (close socket, etc.)
                    self.httpd.server_close()
                    sys.exit(0)
        def stop(self):
            self.httpd.server_close()

    numThreads = 5
    threadArr = []
    for i in range(numThreads):
        threadArr.append(Thread(i))
    while 1:
        try:
            time.sleep(2000)
        except KeyboardInterrupt:
            for i in range(numThreads):
                threadArr[i].stop()
            sys.exit(0)

if __name__ == '__main__':
    # total arguments
    argc = len(sys.argv)

    if argc<2:
        print("Usage: " + sys.argv[0] + " model_file_q4_0.bin [port]")
        exit()
    if argc>=3:
        port = int(sys.argv[2])

    if not os.path.exists(sys.argv[1]):
        print("Cannot find model file: " + sys.argv[1])
        exit()

    mdl_nparts = 1
    for n in range(1,9):
        if os.path.exists(sys.argv[1]+"."+str(n)):
            mdl_nparts += 1
    modelname = os.path.abspath(sys.argv[1])
    print("Loading model: " + modelname)
    loadok = load_model(modelname,128,maxctx,4,mdl_nparts)
    print("Load Model OK: " + str(loadok))

    if loadok:
        print("Starting Kobold HTTP Server on port " + str(port))
        print("Please connect to custom endpoint at http://localhost:"+str(port))
        RunServerMultiThreaded(port)
       