# A hacky little script from Concedo that exposes llama.cpp function bindings
# allowing it to be used via a simulated kobold api endpoint
# generation delay scales linearly with original prompt length.

import ctypes
import os
import argparse
import json, sys, time, asyncio, socket
from aiohttp import web
from concurrent.futures import ThreadPoolExecutor

stop_token_max = 10

class load_model_inputs(ctypes.Structure):
    _fields_ = [("threads", ctypes.c_int),
                ("blasthreads", ctypes.c_int),
                ("max_context_length", ctypes.c_int),
                ("batch_size", ctypes.c_int),
                ("f16_kv", ctypes.c_bool),
                ("executable_path", ctypes.c_char_p),
                ("model_filename", ctypes.c_char_p),
                ("lora_filename", ctypes.c_char_p),
                ("use_mmap", ctypes.c_bool),
                ("use_mlock", ctypes.c_bool),
                ("use_smartcontext", ctypes.c_bool),
                ("unban_tokens", ctypes.c_bool),
                ("clblast_info", ctypes.c_int),
                ("blasbatchsize", ctypes.c_int),
                ("debugmode", ctypes.c_bool),
                ("forceversion", ctypes.c_int),
                ("gpulayers", ctypes.c_int)]

class generation_inputs(ctypes.Structure):
    _fields_ = [("seed", ctypes.c_int),
                ("prompt", ctypes.c_char_p),
                ("max_context_length", ctypes.c_int),
                ("max_length", ctypes.c_int),
                ("temperature", ctypes.c_float),
                ("top_k", ctypes.c_int),
                ("top_a", ctypes.c_float),
                ("top_p", ctypes.c_float),
                ("typical_p", ctypes.c_float),
                ("tfs", ctypes.c_float),
                ("rep_pen", ctypes.c_float),
                ("rep_pen_range", ctypes.c_int),
                ("mirostat", ctypes.c_int),
                ("mirostat_tau", ctypes.c_float),
                ("mirostat_eta", ctypes.c_float),
                ("stop_sequence", ctypes.c_char_p * stop_token_max)]

class generation_outputs(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int),
                ("text", ctypes.c_char * 16384)]

handle = None

def getdirpath():
    return os.path.dirname(os.path.realpath(__file__))
def file_exists(filename):
    return os.path.exists(os.path.join(getdirpath(), filename))

def pick_existant_file(ntoption,nonntoption):
    ntexist = file_exists(ntoption)
    nonntexist = file_exists(nonntoption)
    if os.name == 'nt':
        if nonntexist and not ntexist:
            return nonntoption
        return ntoption
    else:
        if ntexist and not nonntexist:
            return ntoption
        return nonntoption

lib_default = pick_existant_file("koboldcpp.dll","koboldcpp.so")
lib_failsafe = pick_existant_file("koboldcpp_failsafe.dll","koboldcpp_failsafe.so")
lib_openblas = pick_existant_file("koboldcpp_openblas.dll","koboldcpp_openblas.so")
lib_openblas_noavx2 = pick_existant_file("koboldcpp_openblas_noavx2.dll","koboldcpp_openblas_noavx2.so")
lib_clblast = pick_existant_file("koboldcpp_clblast.dll","koboldcpp_clblast.so")


def init_library():
    global handle
    global lib_default,lib_failsafe,lib_openblas,lib_openblas_noavx2,lib_clblast

    libname = ""
    use_blas = False # if true, uses OpenBLAS for acceleration. libopenblas.dll must exist in the same dir.
    use_clblast = False #uses CLBlast instead
    use_noavx2 = False #uses openblas with no avx2 instructions

    if args.noavx2:
        use_noavx2 = True
        if not file_exists(lib_openblas_noavx2) or (os.name=='nt' and not file_exists("libopenblas.dll")):
            print("Warning: OpenBLAS library file not found. Non-BLAS library will be used.")
        elif args.noblas:
            print("!!! Attempting to use FAILSAFE MODE !!!")
        else:
            use_blas = True
            print("Attempting to use non-avx2 compatibility library with OpenBLAS. A compatible libopenblas will be required.")
    elif args.useclblast:
        if not file_exists(lib_clblast) or (os.name=='nt' and not file_exists("clblast.dll")):
            print("Warning: CLBlast library file not found. Non-BLAS library will be used.")
        else:
            print("Attempting to use CLBlast library for faster prompt ingestion. A compatible clblast will be required.")
            use_clblast = True
    else:
        if not file_exists(lib_openblas) or (os.name=='nt' and not file_exists("libopenblas.dll")):
            print("Warning: OpenBLAS library file not found. Non-BLAS library will be used.")
        elif args.noblas:
            print("Attempting to library without OpenBLAS.")
        else:
            use_blas = True
            print("Attempting to use OpenBLAS library for faster prompt ingestion. A compatible libopenblas will be required.")
            if sys.platform=="darwin":
                print("Mac OSX note: Some people have found Accelerate actually faster than OpenBLAS. To compare, run Koboldcpp with --noblas instead.")

    if use_noavx2:
        if use_blas:
            libname = lib_openblas_noavx2
        else:
            libname = lib_failsafe
    else:
        if use_clblast:
            libname = lib_clblast
        elif use_blas:
            libname = lib_openblas
        else:
            libname = lib_default

    print("Initializing dynamic library: " + libname)
    dir_path = getdirpath()

    #OpenBLAS should provide about a 2x speedup on prompt ingestion if compatible.
    handle = ctypes.CDLL(os.path.join(dir_path, libname))

    handle.load_model.argtypes = [load_model_inputs]
    handle.load_model.restype = ctypes.c_bool
    handle.generate.argtypes = [generation_inputs, ctypes.c_wchar_p] #apparently needed for osx to work. i duno why they need to interpret it that way but whatever
    handle.generate.restype = generation_outputs
    handle.new_token.restype = ctypes.c_char_p

def load_model(model_filename):
    inputs = load_model_inputs()
    inputs.model_filename = model_filename.encode("UTF-8")
    inputs.lora_filename = args.lora.encode("UTF-8")
    inputs.batch_size = 8
    inputs.max_context_length = maxctx #initial value to use for ctx, can be overwritten
    inputs.threads = args.threads
    inputs.blasthreads = args.blasthreads
    inputs.f16_kv = True
    inputs.use_mmap = (not args.nommap)
    inputs.use_mlock = args.usemlock
    if args.lora and args.lora!="":
        inputs.use_mmap = False
    inputs.use_smartcontext = args.smartcontext
    inputs.unban_tokens = args.unbantokens
    inputs.blasbatchsize = args.blasbatchsize
    inputs.forceversion = args.forceversion
    inputs.gpulayers = args.gpulayers
    clblastids = 0
    if args.useclblast:
        clblastids = 100 + int(args.useclblast[0])*10 + int(args.useclblast[1])
    inputs.clblast_info = clblastids
    inputs.executable_path = (getdirpath()+"/").encode("UTF-8")
    inputs.debugmode = args.debugmode
    ret = handle.load_model(inputs)
    return ret

def generate(prompt,max_length=20, max_context_length=512,temperature=0.8,top_k=120, top_a=0.0 ,top_p=0.85, typical_p=1.0, tfs=1.0 ,rep_pen=1.1,rep_pen_range=128,seed=-1,stop_sequence=[]):
    inputs = generation_inputs()
    outputs = ctypes.create_unicode_buffer(ctypes.sizeof(generation_outputs))
    inputs.prompt = prompt.encode("UTF-8")
    inputs.max_context_length = max_context_length   # this will resize the context buffer if changed
    inputs.max_length = max_length
    inputs.temperature = temperature
    inputs.top_k = top_k
    inputs.top_a = top_a
    inputs.top_p = top_p
    inputs.typical_p = typical_p
    inputs.tfs = tfs
    inputs.rep_pen = rep_pen
    inputs.rep_pen_range = rep_pen_range
    if args.usemirostat and args.usemirostat[0]>0:
        inputs.mirostat = int(args.usemirostat[0])
        inputs.mirostat_tau = float(args.usemirostat[1])
        inputs.mirostat_eta = float(args.usemirostat[2])
    else:
        inputs.mirostat = inputs.mirostat_tau = inputs.mirostat_eta = 0
    inputs.seed = seed
    for n in range(stop_token_max):
        if not stop_sequence or n >= len(stop_sequence):
            inputs.stop_sequence[n] = "".encode("UTF-8")
        else:
            inputs.stop_sequence[n] = stop_sequence[n].encode("UTF-8")
    ret = handle.generate(inputs,outputs)
    if(ret.status==1):
        return ret.text.decode("UTF-8","ignore")
    return ""

def utfprint(str):
    try:
        print(str)
    except UnicodeEncodeError:
        # Replace or omit the problematic character
        utf_string = str.encode('ascii', 'ignore').decode('ascii')
        print(utf_string)

#################################################################
### A hacky simple HTTP server simulating a kobold api by Concedo
### we are intentionally NOT using flask, because we want MINIMAL dependencies
#################################################################
friendlymodelname = "concedo/koboldcpp"  # local kobold api apparently needs a hardcoded known HF model name
maxctx = 2048
maxlen = 256
modelbusy = False
defaultport = 5001
KcppVersion = "1.29"

class ServerRequestHandler:
    sys_version = ""
    server_version = "ConcedoLlamaForKoboldServer"
    app = web.Application()

    def __init__(self, addr, port, embedded_kailite):
        self.addr = addr
        self.port = port
        self.embedded_kailite = embedded_kailite


    async def generate_text(self, newprompt, genparams, basic_api_flag):
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor()

        def run_blocking():
            # Reset finished status before generating
            handle.bind_set_stream_finished(False)

            if basic_api_flag:
                return generate(
                    prompt=newprompt,
                    max_length=genparams.get('max', 50),
                    temperature=genparams.get('temperature', 0.8),
                    top_k=int(genparams.get('top_k', 120)),
                    top_a=genparams.get('top_a', 0.0),
                    top_p=genparams.get('top_p', 0.85),
                    typical_p=genparams.get('typical', 1.0),
                    tfs=genparams.get('tfs', 1.0),
                    rep_pen=genparams.get('rep_pen', 1.1),
                    rep_pen_range=genparams.get('rep_pen_range', 128),
                    seed=genparams.get('sampler_seed', -1),
                    stop_sequence=genparams.get('stop_sequence', [])
                )

            return generate(prompt=newprompt,
                max_context_length=genparams.get('max_context_length', maxctx),
                max_length=genparams.get('max_length', 50),
                temperature=genparams.get('temperature', 0.8),
                top_k=genparams.get('top_k', 120),
                top_a=genparams.get('top_a', 0.0),
                top_p=genparams.get('top_p', 0.85),
                typical_p=genparams.get('typical', 1.0),
                tfs=genparams.get('tfs', 1.0),
                rep_pen=genparams.get('rep_pen', 1.1),
                rep_pen_range=genparams.get('rep_pen_range', 128),
                seed=genparams.get('sampler_seed', -1),
                stop_sequence=genparams.get('stop_sequence', [])
            )

        recvtxt = await loop.run_in_executor(executor, run_blocking)

        utfprint("\nOutput: " + recvtxt)

        res = {"data": {"seqs":[recvtxt]}} if basic_api_flag else {"results": [{"text": recvtxt}]}

        try:
            return res
        except Exception as e:
            print(f"Generate: Error while generating {e}")


    async def send_sse_event(self, response, event, data):
        await response.write(f'event: {event}\n'.encode())
        await response.write(f'data: {data}\n\n'.encode())

    async def handle_sse_stream(self, request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)

        while not handle.has_finished():
            if not handle.is_locked():
                token = ctypes.string_at(handle.new_token()).decode('utf-8')
                event_data = {"token": token}
                event_str = json.dumps(event_data)
                await self.send_sse_event(response, "message", event_str)

            await asyncio.sleep(0)

        await response.write_eof()
        await response.force_close()

    async def handle_request(self, request, genparams, newprompt, basic_api_flag, stream_flag):
        tasks = []

        if stream_flag:
            tasks.append(self.handle_sse_stream(request,))

        generate_task = asyncio.create_task(self.generate_text(newprompt, genparams, basic_api_flag))
        tasks.append(generate_task)

        try:
            await asyncio.gather(*tasks)
            generate_result = generate_task.result()
            return generate_result
        except Exception as e:
            print(e)


    async def handle_get(self, request):
        global maxctx, maxlen, friendlymodelname, KcppVersion, streamLock
        path = request.path.rstrip('/')

        if path in ["", "/?"] or path.startswith(('/?', '?')):
            if args.stream and not "streaming=1" in path:
                path = path.replace("streaming=0", "")
                if path.startswith(('/?', '?')):
                    path += "&streaming=1"
                else:
                    path = path + "?streaming=1"

            if self.embedded_kailite is None:
                return web.Response(body=f"Embedded Kobold Lite is not found.<br>You will have to connect via the main KoboldAI client, or <a href='https://lite.koboldai.net?local=1&port={self.port}'>use this URL</a> to connect.".encode())
            else:
                return web.Response(body=self.embedded_kailite, content_type='text/html')

        elif path.endswith(('/api/v1/model', '/api/latest/model')):
            return web.json_response({'result': friendlymodelname})

        elif path.endswith(('/api/v1/config/max_length', '/api/latest/config/max_length')):
            return web.json_response({"value": maxlen})

        elif path.endswith(('/api/v1/config/max_context_length', '/api/latest/config/max_context_length')):
            return web.json_response({"value": maxctx})

        elif path.endswith(('/api/v1/config/soft_prompt', '/api/latest/config/soft_prompt')):
            return web.json_response({"value": ""})

        elif path.endswith(('/api/v1/config/soft_prompts_list', '/api/latest/config/soft_prompts_list')):
            return web.json_response({"values": []})

        elif path.endswith(('/api/v1/info/version', '/api/latest/info/version')):
            return web.json_response({"result": "1.2.2"})

        elif path.endswith(('/api/extra/version')):
            return web.json_response({"result": "KoboldCpp", "version": KcppVersion})

        return web.Response(status=404, text="Error: HTTP Server is running, but this endpoint does not exist. Please check the URL.")

    async def handle_post(self, request):
        global modelbusy
        body = await request.content.read()
        basic_api_flag = False
        kai_api_flag = False
        kai_sse_stream_flag = False
        path = request.path.rstrip('/')

        if modelbusy:
            return web.json_response(
                {"detail": {"msg": "Server is busy; please try again later.", "type": "service_unavailable"}},
                status=503,
            )

        if path.endswith('/request'):
            basic_api_flag = True

        if path.endswith(('/api/v1/generate', '/api/latest/generate')):
            kai_api_flag = True

        if path.endswith('/api/extra/generate/stream'):
            kai_api_flag = True
            kai_sse_stream_flag = True

        if basic_api_flag or kai_api_flag:
            genparams = None
            try:
                genparams = json.loads(body)
            except ValueError as e:
                return web.Response(status=503)

            utfprint("\nInput: " + json.dumps(genparams))

            modelbusy = True
            if kai_api_flag:
                fullprompt = genparams.get('prompt', "")
            else:
                fullprompt = genparams.get('text', "")
            newprompt = fullprompt

            gen = await self.handle_request(request, genparams, newprompt, basic_api_flag, kai_sse_stream_flag)

            modelbusy = False

            if not kai_sse_stream_flag:
                return web.Response(body=json.dumps(gen).encode())
            return web.Response();

        return web.Response(status=404)

    async def handle_options(self):
        return web.Response()

    async def handle_head(self):
        return web.Response()

    async def start_server(self):

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.addr, self.port))
        sock.listen(5)

        self.app.router.add_route('GET', '/{tail:.*}', self.handle_get)
        self.app.router.add_route('POST', '/{tail:.*}', self.handle_post)
        self.app.router.add_route('OPTIONS', '/', self.handle_options)
        self.app.router.add_route('HEAD', '/', self.handle_head)

        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.SockSite(runner, sock)
        await site.start()

        # Keep Alive
        try:
            while True:
                await asyncio.sleep(3600)
        except KeyboardInterrupt:
            await runner.cleanup()
            await site.stop()
            await sys.exit(0)
        finally:
            await runner.cleanup()
            await site.stop()
            await sys.exit(0)

async def run_server(addr, port, embedded_kailite=None):
    handler = ServerRequestHandler(addr, port, embedded_kailite)
    await handler.start_server()


def show_gui():
    import tkinter as tk
    from tkinter.filedialog import askopenfilename

    if len(sys.argv) == 1:
        #no args passed at all. Show nooby gui
        root = tk.Tk()
        launchclicked = False

        def guilaunch():
            nonlocal launchclicked
            launchclicked = True
            root.destroy()
            pass

        # Adjust size
        root.geometry("480x360")
        root.title("KoboldCpp v"+KcppVersion)
        root.grid_columnconfigure(0, weight=1)
        tk.Label(root, text = "KoboldCpp Easy Launcher",
                font = ("Arial", 12)).grid(row=0,column=0)
        tk.Label(root, text = "(Note: KoboldCpp only works with GGML model formats!)",
                font = ("Arial", 9)).grid(row=1,column=0)

        blasbatchopts = ["Don't Batch BLAS","BLAS = 32","BLAS = 64","BLAS = 128","BLAS = 256","BLAS = 512","BLAS = 1024"]
        blaschoice = tk.StringVar()
        blaschoice.set("BLAS = 512")

        runopts = ["Use OpenBLAS","Use CLBLast GPU #1","Use CLBLast GPU #2","Use CLBLast GPU #3","Use No BLAS","Use OpenBLAS (Old CPU, noavx2)","Failsafe Mode (Old CPU, noavx)"]
        runchoice = tk.StringVar()
        runchoice.set("Use OpenBLAS")

        def onDropdownChange(event):
            sel = runchoice.get()
            if sel==runopts[1] or sel==runopts[2] or sel==runopts[3]:
                frameC.grid(row=4,column=0,pady=4)
            else:
                frameC.grid_forget()

        frameA = tk.Frame(root)
        tk.OptionMenu( frameA , runchoice , command = onDropdownChange ,*runopts ).grid(row=0,column=0)
        tk.OptionMenu( frameA , blaschoice ,*blasbatchopts ).grid(row=0,column=1)
        frameA.grid(row=2,column=0)

        frameB = tk.Frame(root)
        threads_var=tk.StringVar()
        threads_var.set(str(default_threads))
        threads_lbl = tk.Label(frameB, text = 'Threads: ', font=('calibre',10, 'bold'))
        threads_input = tk.Entry(frameB,textvariable = threads_var, font=('calibre',10,'normal'))
        threads_lbl.grid(row=0,column=0)
        threads_input.grid(row=0,column=1)
        frameB.grid(row=3,column=0,pady=4)

        frameC = tk.Frame(root)
        gpu_layers_var=tk.StringVar()
        gpu_layers_var.set("0")
        gpu_lbl = tk.Label(frameC, text = 'GPU Layers (CLBlast only): ', font=('calibre',10, 'bold'))
        gpu_layers_input = tk.Entry(frameC,textvariable = gpu_layers_var, font=('calibre',10,'normal'))
        gpu_lbl.grid(row=0,column=0)
        gpu_layers_input.grid(row=0,column=1)
        frameC.grid(row=4,column=0,pady=4)
        onDropdownChange(None)

        stream = tk.IntVar()
        smartcontext = tk.IntVar()
        launchbrowser = tk.IntVar(value=1)
        unbantokens = tk.IntVar()
        highpriority = tk.IntVar()
        disablemmap = tk.IntVar()
        frm3 = tk.Frame(root)
        tk.Checkbutton(frm3, text='Streaming Mode',variable=stream, onvalue=1, offvalue=0).grid(row=0,column=0)
        tk.Checkbutton(frm3, text='Use SmartContext',variable=smartcontext, onvalue=1, offvalue=0).grid(row=0,column=1)
        tk.Checkbutton(frm3, text='High Priority',variable=highpriority, onvalue=1, offvalue=0).grid(row=1,column=0)
        tk.Checkbutton(frm3, text='Disable MMAP',variable=disablemmap, onvalue=1, offvalue=0).grid(row=1,column=1)
        tk.Checkbutton(frm3, text='Unban Tokens',variable=unbantokens, onvalue=1, offvalue=0).grid(row=2,column=0)
        tk.Checkbutton(frm3, text='Launch Browser',variable=launchbrowser, onvalue=1, offvalue=0).grid(row=2,column=1)

        frm3.grid(row=5,column=0,pady=4)

        # Create button, it will change label text
        tk.Button( root , text = "Launch", font = ("Impact", 18), bg='#54FA9B', command = guilaunch ).grid(row=6,column=0)
        tk.Label(root, text = "(Please use the Command Line for more advanced options)",
                font = ("Arial", 9)).grid(row=7,column=0)

        root.mainloop()

        if launchclicked==False:
            print("Exiting by user request.")
            time.sleep(2)
            sys.exit()

        #load all the vars
        args.threads = int(threads_var.get())
        args.gpulayers = int(gpu_layers_var.get())

        args.stream = (stream.get()==1)
        args.smartcontext = (smartcontext.get()==1)
        args.launch = (launchbrowser.get()==1)
        args.unbantokens = (unbantokens.get()==1)
        args.highpriority = (highpriority.get()==1)
        args.nommap = (disablemmap.get()==1)
        selrunchoice = runchoice.get()
        selblaschoice = blaschoice.get()

        if selrunchoice==runopts[1]:
            args.useclblast = [0,0]
        if selrunchoice==runopts[2]:
            args.useclblast = [1,0]
        if selrunchoice==runopts[3]:
            args.useclblast = [0,1]
        if selrunchoice==runopts[4]:
            args.noblas = True
        if selrunchoice==runopts[5]:
            args.noavx2 = True
        if selrunchoice==runopts[6]:
            args.noavx2 = True
            args.noblas = True
            args.nommap = True
            print("[Failsafe Mode : mmap is disabled.]")

        if selblaschoice==blasbatchopts[0]:
            args.blasbatchsize = -1
        if selblaschoice==blasbatchopts[1]:
            args.blasbatchsize = 32
        if selblaschoice==blasbatchopts[2]:
            args.blasbatchsize = 64
        if selblaschoice==blasbatchopts[3]:
            args.blasbatchsize = 128
        if selblaschoice==blasbatchopts[4]:
            args.blasbatchsize = 256
        if selblaschoice==blasbatchopts[5]:
            args.blasbatchsize = 512
        if selblaschoice==blasbatchopts[6]:
            args.blasbatchsize = 1024

        root = tk.Tk()
        root.attributes("-alpha", 0)
        args.model_param = askopenfilename(title="Select ggml model .bin files")
        root.destroy()
        if not args.model_param:
            print("\nNo ggml model file was selected. Exiting.")
            time.sleep(2)
            sys.exit(2)

    else:
        root = tk.Tk() #we dont want the useless window to be visible, but we want it in taskbar
        root.attributes("-alpha", 0)
        args.model_param = askopenfilename(title="Select ggml model .bin files")
        root.destroy()
        if not args.model_param:
            print("\nNo ggml model file was selected. Exiting.")
            time.sleep(2)
            sys.exit(2)

def main(args):

    embedded_kailite = None
    if not args.model_param:
        args.model_param = args.model
    if not args.model_param:
        #give them a chance to pick a file
        print("For command line arguments, please refer to --help")
        print("Otherwise, please manually select ggml file:")
        try:
            show_gui()
        except Exception as ex:
            print("File selection GUI unsupported. Please check command line: script.py --help")
            print("Reason for no GUI: " + str(ex))
            time.sleep(2)
            sys.exit(2)

    if args.hordeconfig and args.hordeconfig[0]!="":
        global friendlymodelname, maxlen
        friendlymodelname = "koboldcpp/"+args.hordeconfig[0]
        if len(args.hordeconfig) > 1:
            maxlen = int(args.hordeconfig[1])

    if args.highpriority:
        print("Setting process to Higher Priority - Use Caution")
        try:
            import psutil
            os_used = sys.platform
            process = psutil.Process(os.getpid())  # Set high priority for the python script for the CPU
            oldprio = process.nice()
            if os_used == "win32":  # Windows (either 32-bit or 64-bit)
                process.nice(psutil.REALTIME_PRIORITY_CLASS)
                print("High Priority for Windows Set: " + str(oldprio) + " to " + str(process.nice()))
            elif os_used == "linux":  # linux
                process.nice(psutil.IOPRIO_CLASS_RT)
                print("High Priority for Linux Set: " + str(oldprio) + " to " + str(process.nice()))
            else:  # MAC OS X or other
                process.nice(-18)
                print("High Priority for Other OS Set :" + str(oldprio) + " to " + str(process.nice()))
        except Exception as ex:
             print("Error, Could not change process priority: " + str(ex))

    if args.contextsize:
        global maxctx
        maxctx = args.contextsize

    init_library() # Note: if blas does not exist and is enabled, program will crash.
    print("==========")
    time.sleep(1)
    if not os.path.exists(args.model_param):
        print(f"Cannot find model file: {args.model_param}")
        time.sleep(2)
        sys.exit(2)

    if args.lora and args.lora!="":
        if not os.path.exists(args.lora):
            print(f"Cannot find lora file: {args.lora}")
            time.sleep(2)
            sys.exit(2)
        else:
            args.lora = os.path.abspath(args.lora)

    if args.psutil_set_threads:
        import psutil
        args.threads = psutil.cpu_count(logical=False)
        print("Overriding thread count, using " + str(args.threads) + " threads instead.")

    if not args.blasthreads or args.blasthreads <= 0:
        args.blasthreads = args.threads

    modelname = os.path.abspath(args.model_param)
    print(f"Loading model: {modelname} \n[Threads: {args.threads}, BlasThreads: {args.blasthreads}, SmartContext: {args.smartcontext}]")
    loadok = load_model(modelname)
    print("Load Model OK: " + str(loadok))

    if not loadok:
        print("Could not load model: " + modelname)
        time.sleep(2)
        sys.exit(3)
    try:
        basepath = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(basepath, "klite.embd"), mode='rb') as f:
            embedded_kailite = f.read()
            print("Embedded Kobold Lite loaded.")
    except:
        print("Could not find Kobold Lite. Embedded Kobold Lite will not be available.")

    if args.port_param!=defaultport:
        args.port = args.port_param
    print(f"Starting Kobold HTTP Server on port {args.port}")
    epurl = ""
    if args.host=="":
        epurl = f"http://localhost:{args.port}"
    else:
        epurl = f"http://{args.host}:{args.port}"

    if args.launch:
        try:
            import webbrowser as wb
            wb.open(epurl)
        except:
            print("--launch was set, but could not launch web browser automatically.")
    print(f"Please connect to custom endpoint at {epurl}")
    asyncio.run(run_server(args.host, args.port, embedded_kailite))

if __name__ == '__main__':
    print("Welcome to KoboldCpp - Version " + KcppVersion) # just update version manually
    # print("Python version: " + sys.version)
    parser = argparse.ArgumentParser(description='KoboldCpp Server')
    modelgroup = parser.add_mutually_exclusive_group() #we want to be backwards compatible with the unnamed positional args
    modelgroup.add_argument("--model", help="Model file to load", nargs="?")
    modelgroup.add_argument("model_param", help="Model file to load (positional)", nargs="?")
    portgroup = parser.add_mutually_exclusive_group() #we want to be backwards compatible with the unnamed positional args
    portgroup.add_argument("--port", help="Port to listen on", default=defaultport, type=int, action='store')
    portgroup.add_argument("port_param", help="Port to listen on (positional)", default=defaultport, nargs="?", type=int, action='store')
    parser.add_argument("--host", help="Host IP to listen on. If empty, all routable interfaces are accepted.", default="")
    parser.add_argument("--launch", help="Launches a web browser when load is completed.", action='store_true')
    parser.add_argument("--lora", help="LLAMA models only, applies a lora file on top of model. Experimental.", default="")
    physical_core_limit = 1
    if os.cpu_count()!=None and os.cpu_count()>1:
        physical_core_limit = int(os.cpu_count()/2)
    default_threads = (physical_core_limit if physical_core_limit<=3 else max(3,physical_core_limit-1))
    parser.add_argument("--threads", help="Use a custom number of threads if specified. Otherwise, uses an amount based on CPU cores", type=int, default=default_threads)
    parser.add_argument("--blasthreads", help="Use a different number of threads during BLAS if specified. Otherwise, has the same value as --threads",metavar=('[threads]'), type=int, default=0)
    parser.add_argument("--psutil_set_threads", help="Experimental flag. If set, uses psutils to determine thread count based on physical cores.", action='store_true')
    parser.add_argument("--highpriority", help="Experimental flag. If set, increases the process CPU priority, potentially speeding up generation. Use caution.", action='store_true')
    parser.add_argument("--contextsize", help="Controls the memory allocated for maximum context size, only change if you need more RAM for big contexts. (default 2048)", type=int,choices=[512,1024,2048,4096,8192], default=2048)
    parser.add_argument("--blasbatchsize", help="Sets the batch size used in BLAS processing (default 512). Setting it to -1 disables BLAS mode, but keeps other benefits like GPU offload.", type=int,choices=[-1,32,64,128,256,512,1024], default=512)
    parser.add_argument("--stream", help="Uses pseudo streaming when generating tokens. Only for the Kobold Lite UI.", action='store_true')
    parser.add_argument("--smartcontext", help="Reserving a portion of context to try processing less frequently.", action='store_true')
    parser.add_argument("--unbantokens", help="Normally, KoboldAI prevents certain tokens such as EOS and Square Brackets. This flag unbans them.", action='store_true')
    parser.add_argument("--usemirostat", help="Experimental! Replaces your samplers with mirostat. Takes 3 params = [type(0/1/2), tau(5.0), eta(0.1)].",metavar=('[type]', '[tau]', '[eta]'), type=float, nargs=3)
    parser.add_argument("--forceversion", help="If the model file format detection fails (e.g. rogue modified model) you can set this to override the detected format (enter desired version, e.g. 401 for GPTNeoX-Type2).",metavar=('[version]'), type=int, default=0)
    parser.add_argument("--nommap", help="If set, do not use mmap to load newer models", action='store_true')
    parser.add_argument("--usemlock", help="For Apple Systems. Force system to keep model in RAM rather than swapping or compressing", action='store_true')
    parser.add_argument("--noavx2", help="Do not use AVX2 instructions, a slower compatibility mode for older devices. Does not work with --clblast.", action='store_true')
    parser.add_argument("--debugmode", help="Shows additional debug info in the terminal.", action='store_true')
    parser.add_argument("--skiplauncher", help="Doesn't display or use the new GUI launcher.", action='store_true')
    parser.add_argument("--hordeconfig", help="Sets the display model name to something else, for easy use on AI Horde. An optional second parameter sets the horde max gen length.",metavar=('[hordename]', '[hordelength]'), nargs='+')
    compatgroup = parser.add_mutually_exclusive_group()
    compatgroup.add_argument("--noblas", help="Do not use OpenBLAS for accelerated prompt ingestion", action='store_true')
    compatgroup.add_argument("--useclblast", help="Use CLBlast instead of OpenBLAS for prompt ingestion. Must specify exactly 2 arguments, platform ID and device ID (e.g. --useclblast 1 0).", type=int, choices=range(0,9), nargs=2)
    parser.add_argument("--gpulayers", help="Set number of layers to offload to GPU when using CLBlast. Requires CLBlast.",metavar=('[GPU layers]'), type=int, default=0)
    args = parser.parse_args()
    main(args)
