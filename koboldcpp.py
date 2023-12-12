#!/usr/bin/env python3
#-*- coding: utf-8 -*-

# KoboldCpp is an easy-to-use AI text-generation software for GGML models.
# It's a single self contained distributable from Concedo, that builds off llama.cpp,
# and adds a versatile Kobold API endpoint, additional format support,
# backward compatibility, as well as a fancy UI with persistent stories,
# editing tools, save formats, memory, world info, author's note, characters,
# scenarios and everything Kobold and Kobold Lite have to offer.

import ctypes
import os
import argparse
import json, sys, http.server, time, asyncio, socket, threading
from concurrent.futures import ThreadPoolExecutor

sampler_order_max = 7
stop_token_max = 16
ban_token_max = 16
tensor_split_max = 16

class load_model_inputs(ctypes.Structure):
    _fields_ = [("threads", ctypes.c_int),
                ("blasthreads", ctypes.c_int),
                ("max_context_length", ctypes.c_int),
                ("batch_size", ctypes.c_int),
                ("low_vram", ctypes.c_bool),
                ("use_mmq", ctypes.c_bool),
                ("executable_path", ctypes.c_char_p),
                ("model_filename", ctypes.c_char_p),
                ("lora_filename", ctypes.c_char_p),
                ("lora_base", ctypes.c_char_p),
                ("use_mmap", ctypes.c_bool),
                ("use_mlock", ctypes.c_bool),
                ("use_smartcontext", ctypes.c_bool),
                ("use_contextshift", ctypes.c_bool),
                ("clblast_info", ctypes.c_int),
                ("cublas_info", ctypes.c_int),
                ("blasbatchsize", ctypes.c_int),
                ("debugmode", ctypes.c_int),
                ("forceversion", ctypes.c_int),
                ("gpulayers", ctypes.c_int),
                ("rope_freq_scale", ctypes.c_float),
                ("rope_freq_base", ctypes.c_float),
                ("banned_tokens", ctypes.c_char_p * ban_token_max),
                ("tensor_split", ctypes.c_float * tensor_split_max)]

class generation_inputs(ctypes.Structure):
    _fields_ = [("seed", ctypes.c_int),
                ("prompt", ctypes.c_char_p),
                ("memory", ctypes.c_char_p),
                ("max_context_length", ctypes.c_int),
                ("max_length", ctypes.c_int),
                ("temperature", ctypes.c_float),
                ("top_k", ctypes.c_int),
                ("top_a", ctypes.c_float),
                ("top_p", ctypes.c_float),
                ("min_p", ctypes.c_float),
                ("typical_p", ctypes.c_float),
                ("tfs", ctypes.c_float),
                ("rep_pen", ctypes.c_float),
                ("rep_pen_range", ctypes.c_int),
                ("mirostat", ctypes.c_int),
                ("mirostat_tau", ctypes.c_float),
                ("mirostat_eta", ctypes.c_float),
                ("sampler_order", ctypes.c_int * sampler_order_max),
                ("sampler_len", ctypes.c_int),
                ("unban_tokens_rt", ctypes.c_bool),
                ("stop_sequence", ctypes.c_char_p * stop_token_max),
                ("stream_sse", ctypes.c_bool),
                ("grammar", ctypes.c_char_p),
                ("grammar_retain_state", ctypes.c_bool),
                ("quiet", ctypes.c_bool)]

class generation_outputs(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int),
                ("text", ctypes.c_char * 32768)]

class token_count_outputs(ctypes.Structure):
    _fields_ = [("count", ctypes.c_int),
                ("ids", ctypes.POINTER(ctypes.c_int))]

handle = None

def getdirpath():
    return os.path.dirname(os.path.realpath(__file__))
def getabspath():
    return os.path.dirname(os.path.abspath(__file__))
def file_exists(filename):
    return os.path.exists(os.path.join(getdirpath(), filename))

def pick_existant_file(ntoption,nonntoption):
    precompiled_prefix = "precompiled_"
    ntexist = file_exists(ntoption)
    nonntexist = file_exists(nonntoption)
    precompiled_ntexist = file_exists(precompiled_prefix+ntoption)
    precompiled_nonntexist = file_exists(precompiled_prefix+nonntoption)
    if os.name == 'nt':
        if not ntexist and precompiled_ntexist:
            return (precompiled_prefix+ntoption)
        if nonntexist and not ntexist:
            return nonntoption
        return ntoption
    else:
        if not nonntexist and precompiled_nonntexist:
            return (precompiled_prefix+nonntoption)
        if ntexist and not nonntexist:
            return ntoption
        return nonntoption

lib_default = pick_existant_file("koboldcpp_default.dll","koboldcpp_default.so")
lib_failsafe = pick_existant_file("koboldcpp_failsafe.dll","koboldcpp_failsafe.so")
lib_openblas = pick_existant_file("koboldcpp_openblas.dll","koboldcpp_openblas.so")
lib_noavx2 = pick_existant_file("koboldcpp_noavx2.dll","koboldcpp_noavx2.so")
lib_clblast = pick_existant_file("koboldcpp_clblast.dll","koboldcpp_clblast.so")
lib_clblast_noavx2 = pick_existant_file("koboldcpp_clblast_noavx2.dll","koboldcpp_clblast_noavx2.so")
lib_cublas = pick_existant_file("koboldcpp_cublas.dll","koboldcpp_cublas.so")
lib_hipblas = pick_existant_file("koboldcpp_hipblas.dll","koboldcpp_hipblas.so")


def init_library():
    global handle, args
    global lib_default,lib_failsafe,lib_openblas,lib_noavx2,lib_clblast,lib_clblast_noavx2,lib_cublas,lib_hipblas

    libname = ""
    use_openblas = False # if true, uses OpenBLAS for acceleration. libopenblas.dll must exist in the same dir.
    use_clblast = False #uses CLBlast instead
    use_cublas = False #uses cublas instead
    use_hipblas = False #uses hipblas instead
    use_noavx2 = False #uses no avx2 instructions
    use_failsafe = False #uses no intrinsics, failsafe mode
    if args.noavx2:
        use_noavx2 = True
        if args.useclblast:
            if not file_exists(lib_clblast_noavx2) or (os.name=='nt' and not file_exists("clblast.dll")):
                print("Warning: NoAVX2 CLBlast library file not found. Non-BLAS library will be used.")
            else:
                print("Attempting to use NoAVX2 CLBlast library for faster prompt ingestion. A compatible clblast will be required.")
                use_clblast = True
        else:
            if not file_exists(lib_noavx2):
                print("Warning: NoAVX2 library file not found. Failsafe library will be used.")
            elif (args.noblas and args.nommap):
                use_failsafe = True
                print("!!! Attempting to use FAILSAFE MODE !!!")
            else:
                print("Attempting to use non-avx2 compatibility library.")
    elif args.useclblast:
        if not file_exists(lib_clblast) or (os.name=='nt' and not file_exists("clblast.dll")):
            print("Warning: CLBlast library file not found. Non-BLAS library will be used.")
        else:
            print("Attempting to use CLBlast library for faster prompt ingestion. A compatible clblast will be required.")
            use_clblast = True
    elif (args.usecublas is not None):
        if not file_exists(lib_cublas) and not file_exists(lib_hipblas):
            print("Warning: CuBLAS library file not found. Non-BLAS library will be used.")
        else:
            if file_exists(lib_cublas):
                print("Attempting to use CuBLAS library for faster prompt ingestion. A compatible CuBLAS will be required.")
                use_cublas = True
            elif file_exists(lib_hipblas):
                print("Attempting to use hipBLAS library for faster prompt ingestion. A compatible AMD GPU will be required.")
                use_hipblas = True

    else:
        if not file_exists(lib_openblas) or (os.name=='nt' and not file_exists("libopenblas.dll")):
            print("Warning: OpenBLAS library file not found. Non-BLAS library will be used.")
        elif args.noblas:
            print("Attempting to library without OpenBLAS.")
        else:
            use_openblas = True
            print("Attempting to use OpenBLAS library for faster prompt ingestion. A compatible libopenblas will be required.")
            if sys.platform=="darwin":
                print("Mac OSX note: Some people have found Accelerate actually faster than OpenBLAS. To compare, run Koboldcpp with --noblas instead.")

    if use_noavx2:
        if use_failsafe:
            libname = lib_failsafe
        elif use_clblast:
            libname = lib_clblast_noavx2
        else:
            libname = lib_noavx2
    else:
        if use_clblast:
            libname = lib_clblast
        elif use_cublas:
            libname = lib_cublas
        elif use_hipblas:
            libname = lib_hipblas
        elif use_openblas:
            libname = lib_openblas
        else:
            libname = lib_default

    print("Initializing dynamic library: " + libname)
    dir_path = getdirpath()
    abs_path = getabspath()

    #add all potential paths
    if os.name=='nt':
        os.add_dll_directory(dir_path)
        os.add_dll_directory(abs_path)
        os.add_dll_directory(os.getcwd())
        if libname == lib_hipblas and "HIP_PATH" in os.environ:
            os.add_dll_directory(os.path.join(os.environ["HIP_PATH"], "bin"))
            if args.debugmode == 1:
                print(f"HIP/ROCm SDK at {os.environ['HIP_PATH']} included in .DLL load path")
    handle = ctypes.CDLL(os.path.join(dir_path, libname))

    handle.load_model.argtypes = [load_model_inputs]
    handle.load_model.restype = ctypes.c_bool
    handle.generate.argtypes = [generation_inputs, ctypes.c_wchar_p] #apparently needed for osx to work. i duno why they need to interpret it that way but whatever
    handle.generate.restype = generation_outputs
    handle.new_token.restype = ctypes.c_char_p
    handle.new_token.argtypes = [ctypes.c_int]
    handle.get_stream_count.restype = ctypes.c_int
    handle.has_finished.restype = ctypes.c_bool
    handle.get_last_eval_time.restype = ctypes.c_float
    handle.get_last_process_time.restype = ctypes.c_float
    handle.get_last_token_count.restype = ctypes.c_int
    handle.get_total_gens.restype = ctypes.c_int
    handle.get_last_stop_reason.restype = ctypes.c_int
    handle.abort_generate.restype = ctypes.c_bool
    handle.token_count.restype = token_count_outputs
    handle.get_pending_output.restype = ctypes.c_char_p

def load_model(model_filename):
    global args
    inputs = load_model_inputs()
    inputs.model_filename = model_filename.encode("UTF-8")
    inputs.batch_size = 8
    inputs.max_context_length = maxctx #initial value to use for ctx, can be overwritten
    inputs.threads = args.threads
    inputs.low_vram = (True if (args.usecublas and "lowvram" in args.usecublas) else False)
    inputs.use_mmq = (True if (args.usecublas and "mmq" in args.usecublas) else False)
    inputs.blasthreads = args.blasthreads
    inputs.use_mmap = (not args.nommap)
    inputs.use_mlock = args.usemlock
    inputs.lora_filename = "".encode("UTF-8")
    inputs.lora_base = "".encode("UTF-8")
    if args.lora:
        inputs.lora_filename = args.lora[0].encode("UTF-8")
        inputs.use_mmap = False
        if len(args.lora) > 1:
            inputs.lora_base = args.lora[1].encode("UTF-8")
    inputs.use_smartcontext = args.smartcontext
    inputs.use_contextshift = (0 if args.noshift else 1)
    inputs.blasbatchsize = args.blasbatchsize
    inputs.forceversion = args.forceversion
    inputs.gpulayers = args.gpulayers
    inputs.rope_freq_scale = args.ropeconfig[0]
    if len(args.ropeconfig)>1:
        inputs.rope_freq_base = args.ropeconfig[1]
    else:
        inputs.rope_freq_base = 10000
    clblastids = 0
    if args.useclblast:
        clblastids = 100 + int(args.useclblast[0])*10 + int(args.useclblast[1])
    inputs.clblast_info = clblastids

    for n in range(tensor_split_max):
        if args.tensor_split and n < len(args.tensor_split):
            inputs.tensor_split[n] = float(args.tensor_split[n])
        else:
            inputs.tensor_split[n] = 0

    # we must force an explicit tensor split
    # otherwise the default will divide equally and multigpu crap will slow it down badly
    inputs.cublas_info = 0

    if not args.tensor_split:
        if (args.usecublas and "0" in args.usecublas):
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ["HIP_VISIBLE_DEVICES"] = "0"
        elif (args.usecublas and "1" in args.usecublas):
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            os.environ["HIP_VISIBLE_DEVICES"] = "1"
        elif (args.usecublas and "2" in args.usecublas):
            os.environ["CUDA_VISIBLE_DEVICES"] = "2"
            os.environ["HIP_VISIBLE_DEVICES"] = "2"
        elif (args.usecublas and "3" in args.usecublas):
            os.environ["CUDA_VISIBLE_DEVICES"] = "3"
            os.environ["HIP_VISIBLE_DEVICES"] = "3"
    else:
        if (args.usecublas and "0" in args.usecublas):
            inputs.cublas_info = 0
        elif (args.usecublas and "1" in args.usecublas):
            inputs.cublas_info = 1
        elif (args.usecublas and "2" in args.usecublas):
            inputs.cublas_info = 2
        elif (args.usecublas and "3" in args.usecublas):
            inputs.cublas_info = 3

    inputs.executable_path = (getdirpath()+"/").encode("UTF-8")
    inputs.debugmode = args.debugmode
    banned_tokens = args.bantokens
    for n in range(ban_token_max):
        if not banned_tokens or n >= len(banned_tokens):
            inputs.banned_tokens[n] = "".encode("UTF-8")
        else:
            inputs.banned_tokens[n] = banned_tokens[n].encode("UTF-8")
    ret = handle.load_model(inputs)
    return ret

def generate(prompt, memory="", max_length=32, max_context_length=512, temperature=0.7, top_k=100, top_a=0.0, top_p=0.92, min_p=0.0, typical_p=1.0, tfs=1.0, rep_pen=1.1, rep_pen_range=128, mirostat=0, mirostat_tau=5.0, mirostat_eta=0.1, sampler_order=[6,0,1,3,4,2,5], seed=-1, stop_sequence=[], use_default_badwordsids=False, stream_sse=False, grammar='', grammar_retain_state=False, genkey='', trimstop=False, quiet=False):
    global maxctx, args, currentusergenkey, totalgens
    inputs = generation_inputs()
    outputs = ctypes.create_unicode_buffer(ctypes.sizeof(generation_outputs))
    inputs.prompt = prompt.encode("UTF-8")
    inputs.memory = memory.encode("UTF-8")
    if max_length >= max_context_length:
        max_length = max_context_length-1
    inputs.max_context_length = max_context_length   # this will resize the context buffer if changed
    global showmaxctxwarning
    if max_context_length > maxctx:
        if showmaxctxwarning:
            print(f"\n(Warning! Request max_context_length={max_context_length} exceeds allocated context size of {maxctx}. It will be reduced to fit. Consider launching with increased --contextsize to avoid errors. This message will only show once per session.)")
            showmaxctxwarning = False
        max_context_length = maxctx
    inputs.max_length = max_length
    inputs.temperature = temperature
    inputs.top_k = top_k
    inputs.top_a = top_a
    inputs.top_p = top_p
    inputs.min_p = min_p
    inputs.typical_p = typical_p
    inputs.tfs = tfs
    inputs.rep_pen = rep_pen
    inputs.rep_pen_range = rep_pen_range
    inputs.stream_sse = stream_sse
    inputs.quiet = quiet
    inputs.grammar = grammar.encode("UTF-8")
    inputs.grammar_retain_state = grammar_retain_state
    inputs.unban_tokens_rt = not use_default_badwordsids
    if mirostat in (1, 2):
        inputs.mirostat = mirostat
        inputs.mirostat_tau = mirostat_tau
        inputs.mirostat_eta = mirostat_eta
    else:
        inputs.mirostat = inputs.mirostat_tau = inputs.mirostat_eta = 0
    if sampler_order and 0 < len(sampler_order) <= sampler_order_max:
        try:
            for i, sampler in enumerate(sampler_order):
                inputs.sampler_order[i] = sampler
            inputs.sampler_len = len(sampler_order)
            global showsamplerwarning
            if showsamplerwarning and inputs.mirostat==0 and inputs.sampler_len>0 and (inputs.sampler_order[0]!=6 or inputs.sampler_order[inputs.sampler_len-1]!=5):
                print("\n(Note: Sub-optimal sampler_order detected. You may have reduced quality. Recommended sampler values are [6,0,1,3,4,2,5]. This message will only show once per session.)")
                showsamplerwarning = False
        except TypeError as e:
            print("ERROR: sampler_order must be a list of integers: " + str(e))
    inputs.seed = seed
    for n in range(stop_token_max):
        if not stop_sequence or n >= len(stop_sequence):
            inputs.stop_sequence[n] = "".encode("UTF-8")
        else:
            inputs.stop_sequence[n] = stop_sequence[n].encode("UTF-8")
    currentusergenkey = genkey
    totalgens += 1
    ret = handle.generate(inputs,outputs)
    outstr = ""
    if ret.status==1:
        outstr = ret.text.decode("UTF-8","ignore")
    if trimstop:
        for trim_str in stop_sequence:
            sindex = outstr.find(trim_str)
            if sindex != -1 and trim_str!="":
                outstr = outstr[:sindex]
    return outstr

def utfprint(str):
    try:
        print(str)
    except UnicodeEncodeError:
        # Replace or omit the problematic character
        utf_string = str.encode('ascii', 'ignore').decode('ascii')
        utf_string = utf_string.replace('\a', '') #remove bell characters
        print(utf_string)

def bring_terminal_to_foreground():
    if os.name=='nt':
        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 9)
        ctypes.windll.user32.SetForegroundWindow(ctypes.windll.kernel32.GetConsoleWindow())


#################################################################
### A hacky simple HTTP server simulating a kobold api by Concedo
### we are intentionally NOT using flask, because we want MINIMAL dependencies
#################################################################
friendlymodelname = "concedo/koboldcpp"  # local kobold api apparently needs a hardcoded known HF model name
maxctx = 2048
maxhordectx = 2048
maxhordelen = 256
modelbusy = threading.Lock()
requestsinqueue = 0
defaultport = 5001
KcppVersion = "1.52"
showdebug = True
showsamplerwarning = True
showmaxctxwarning = True
session_kudos_earned = 0
session_jobs = 0
session_starttime = None
exitcounter = -1
punishcounter = 0 #causes a timeout if too many errors
rewardcounter = 0 #reduces error counts for successful jobs
totalgens = 0
currentusergenkey = "" #store a special key so polled streaming works even in multiuser
args = None #global args
gui_layers_untouched = True
preloaded_story = None

class ServerRequestHandler(http.server.SimpleHTTPRequestHandler):
    sys_version = ""
    server_version = "ConcedoLlamaForKoboldServer"

    def __init__(self, addr, port, embedded_kailite, embedded_kcpp_docs):
        self.addr = addr
        self.port = port
        self.embedded_kailite = embedded_kailite
        self.embedded_kcpp_docs = embedded_kcpp_docs

    def __call__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        global showdebug
        if showdebug:
            super().log_message(format, *args)
        pass

    async def generate_text(self, genparams, api_format, stream_flag):
        global friendlymodelname
        is_quiet = args.quiet
        def run_blocking(): #api format 1=basic,2=kai,3=oai,4=oai-chat
            if api_format==1:
                genparams["prompt"] = genparams.get('text', "")
                genparams["top_k"] = int(genparams.get('top_k', 120))
                genparams["max_length"] = genparams.get('max', 100)

            elif api_format==3 or api_format==4:
                frqp = genparams.get('frequency_penalty', 0.1)
                scaled_rep_pen = genparams.get('presence_penalty', frqp) + 1
                genparams["max_length"] = genparams.get('max_tokens', 100)
                genparams["rep_pen"] = scaled_rep_pen
                # openai allows either a string or a list as a stop sequence
                if isinstance(genparams.get('stop',[]), list):
                    genparams["stop_sequence"] = genparams.get('stop', [])
                else:
                    genparams["stop_sequence"] = [genparams.get('stop')]

                genparams["sampler_seed"] = genparams.get('seed', -1)
                genparams["use_default_badwordsids"] = genparams.get('ignore_eos', False)
                genparams["mirostat"] = genparams.get('mirostat_mode', 0)

                if api_format==4:
                    # translate openai chat completion messages format into one big string.
                    messages_array = genparams.get('messages', [])
                    adapter_obj = genparams.get('adapter', {})
                    messages_string = ""
                    system_message_start = adapter_obj.get("system_start", "\n### Instruction:\n")
                    system_message_end = adapter_obj.get("system_end", "")
                    user_message_start = adapter_obj.get("user_start", "\n### Instruction:\n")
                    user_message_end = adapter_obj.get("user_end", "")
                    assistant_message_start = adapter_obj.get("assistant_start", "\n### Response:\n")
                    assistant_message_end = adapter_obj.get("assistant_end", "")

                    for message in messages_array:
                        if message['role'] == "system":
                            messages_string += system_message_start
                        elif message['role'] == "user":
                            messages_string += user_message_start
                        elif message['role'] == "assistant":
                            messages_string += assistant_message_start

                        messages_string += message['content']

                        if message['role'] == "system":
                            messages_string += system_message_end
                        elif message['role'] == "user":
                            messages_string += user_message_end
                        elif message['role'] == "assistant":
                            messages_string += assistant_message_end

                    messages_string += assistant_message_start
                    genparams["prompt"] = messages_string

            return generate(
                prompt=genparams.get('prompt', ""),
                memory=genparams.get('memory', ""),
                max_context_length=genparams.get('max_context_length', maxctx),
                max_length=genparams.get('max_length', 100),
                temperature=genparams.get('temperature', 0.7),
                top_k=genparams.get('top_k', 100),
                top_a=genparams.get('top_a', 0.0),
                top_p=genparams.get('top_p', 0.92),
                min_p=genparams.get('min_p', 0.0),
                typical_p=genparams.get('typical', 1.0),
                tfs=genparams.get('tfs', 1.0),
                rep_pen=genparams.get('rep_pen', 1.1),
                rep_pen_range=genparams.get('rep_pen_range', 256),
                mirostat=genparams.get('mirostat', 0),
                mirostat_tau=genparams.get('mirostat_tau', 5.0),
                mirostat_eta=genparams.get('mirostat_eta', 0.1),
                sampler_order=genparams.get('sampler_order', [6,0,1,3,4,2,5]),
                seed=genparams.get('sampler_seed', -1),
                stop_sequence=genparams.get('stop_sequence', []),
                use_default_badwordsids=genparams.get('use_default_badwordsids', False),
                stream_sse=stream_flag,
                grammar=genparams.get('grammar', ''),
                grammar_retain_state = genparams.get('grammar_retain_state', False),
                genkey=genparams.get('genkey', ''),
                trimstop=genparams.get('trim_stop', False),
                quiet=is_quiet)

        recvtxt = ""
        if stream_flag:
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor()
            recvtxt = await loop.run_in_executor(executor, run_blocking)
        else:
            recvtxt = run_blocking()

        if (args.debugmode != -1 and not is_quiet) or args.debugmode >= 1:
            utfprint("\nOutput: " + recvtxt)

        if api_format==1:
            res = {"data": {"seqs":[recvtxt]}}
        elif api_format==3:
            res = {"id": "cmpl-1", "object": "text_completion", "created": 1, "model": friendlymodelname,
            "choices": [{"text": recvtxt, "index": 0, "finish_reason": "length"}]}
        elif api_format==4:
            res = {"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": friendlymodelname,
            "choices": [{"index": 0, "message":{"role": "assistant", "content": recvtxt,}, "finish_reason": "length"}]}
        else:
            res = {"results": [{"text": recvtxt}]}

        try:
            return res
        except Exception as e:
            print(f"Generate: Error while generating: {e}")


    async def send_oai_sse_event(self, data):
        if data=="[DONE]":
            self.wfile.write(f'data: {data}'.encode())
        else:
            self.wfile.write(f'data: {data}\r\n\r\n'.encode())
        self.wfile.flush()

    async def send_kai_sse_event(self, data):
        self.wfile.write(f'event: message\n'.encode())
        self.wfile.write(f'data: {data}\n\n'.encode())
        self.wfile.flush()

    async def handle_sse_stream(self, api_format):
        global friendlymodelname
        self.send_response(200)
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "keep-alive")
        self.end_headers(content_type='text/event-stream')

        current_token = 0
        incomplete_token_buffer = bytearray()
        await asyncio.sleep(0.05) #anti race condition, prevent check from overtaking generate
        while True:
            streamDone = handle.has_finished() #exit next loop on done
            tokenStr = ""
            streamcount = handle.get_stream_count()
            while current_token < streamcount:
                token = handle.new_token(current_token)

                if token is None: # Token isnt ready yet, received nullpointer
                    break

                current_token += 1
                newbyte = ctypes.string_at(token)
                incomplete_token_buffer += bytearray(newbyte)
                tokenSeg = incomplete_token_buffer.decode("UTF-8","ignore")
                if tokenSeg!="":
                    incomplete_token_buffer.clear()
                    tokenStr += tokenSeg

            if tokenStr!="":
                if api_format == 4:  # if oai chat, set format to expected openai streaming response
                    event_str = json.dumps({"id":"koboldcpp","object":"chat.completion.chunk","created":1,"model":friendlymodelname,"choices":[{"index":0,"finish_reason":"length","delta":{'role':'assistant','content':tokenStr}}]})
                    await self.send_oai_sse_event(event_str)
                elif api_format == 3:  # non chat completions
                    event_str = json.dumps({"id":"koboldcpp","object":"text_completion","created":1,"model":friendlymodelname,"choices":[{"index":0,"finish_reason":"length","text":tokenStr}]})
                    await self.send_oai_sse_event(event_str)
                else:
                    event_str = json.dumps({"token": tokenStr})
                    await self.send_kai_sse_event(event_str)
                tokenStr = ""

            else:
                await asyncio.sleep(0.02) #this should keep things responsive

            if streamDone:
                if api_format == 4:  # if oai chat, send last [DONE] message consistent with openai format
                    await self.send_oai_sse_event('[DONE]')
                break

        # flush buffers, sleep a bit to make sure all data sent, and then force close the connection
        self.wfile.flush()
        await asyncio.sleep(0.1)
        self.close_connection = True
        await asyncio.sleep(0.05)


    async def handle_request(self, genparams, api_format, stream_flag):
        tasks = []

        if stream_flag:
            tasks.append(self.handle_sse_stream(api_format))

        generate_task = asyncio.create_task(self.generate_text(genparams, api_format, stream_flag))
        tasks.append(generate_task)

        try:
            await asyncio.gather(*tasks)
            generate_result = generate_task.result()
            return generate_result
        except ConnectionAbortedError as cae: # attempt to abort if connection lost
            print(cae)
            handle.abort_generate()
            time.sleep(0.1) #short delay
        except Exception as e:
            print(e)

    def noscript_webui(self):
        global modelbusy
        import html
        import urllib.parse as urlparse
        parsed_url = urlparse.urlparse(self.path)
        parsed_dict = urlparse.parse_qs(parsed_url.query)
        reply = ""
        status = str(parsed_dict['status'][0]) if 'status' in parsed_dict else "Ready To Generate"
        prompt = str(parsed_dict['prompt'][0]) if 'prompt' in parsed_dict else ""
        max_length = int(parsed_dict['max_length'][0]) if 'max_length' in parsed_dict else 100
        temperature = float(parsed_dict['temperature'][0]) if 'temperature' in parsed_dict else 0.7
        top_k = int(parsed_dict['top_k'][0]) if 'top_k' in parsed_dict else 100
        top_p = float(parsed_dict['top_p'][0]) if 'top_p' in parsed_dict else 0.9
        rep_pen = float(parsed_dict['rep_pen'][0]) if 'rep_pen' in parsed_dict else 1.1
        use_default_badwordsids = int(parsed_dict['use_default_badwordsids'][0]) if 'use_default_badwordsids' in parsed_dict else 0
        gencommand = (parsed_dict['generate'][0] if 'generate' in parsed_dict else "")=="Generate"

        if modelbusy.locked():
            status = "Model is currently busy, try again later."
        elif gencommand:
            if prompt=="" or max_length<=0:
                status = "Need a valid prompt and length to generate."
            else:
                if max_length>512:
                    max_length = 512
                epurl = f"http://localhost:{args.port}"
                if args.host!="":
                    epurl = f"http://{args.host}:{args.port}"
                gen_payload = {"prompt": prompt,"max_length": max_length,"temperature": temperature,"prompt": prompt,"top_k": top_k,"top_p": top_p,"rep_pen": rep_pen,"use_default_badwordsids":use_default_badwordsids}
                respjson = make_url_request(f'{epurl}/api/v1/generate', gen_payload)
                reply = html.escape(respjson["results"][0]["text"])
                status = "Generation Completed"

            if "generate" in parsed_dict:
                del parsed_dict["generate"]
            parsed_dict["prompt"] = prompt + reply
            parsed_dict["status"] = status
            updated_query_string = urlparse.urlencode(parsed_dict, doseq=True)
            updated_path = parsed_url._replace(query=updated_query_string).geturl()
            self.path = updated_path
            self.send_response(302)
            self.send_header("location", self.path)
            self.end_headers(content_type='text/html')
            return

        finalhtml = f'''<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KoboldCpp NoScript Mode</title></head><body>
<h2>KoboldCpp NoScript Mode</h2>
<div>
<p>KoboldCpp can be used without Javascript enabled, however this is not recommended.
<br>If you have Javascript, please use <a href="/">Kobold Lite WebUI</a> instead.</p><hr>
<form action="/noscript">
Enter Prompt:<br>
<textarea name="prompt" cols="60" rows="8" wrap="soft" placeholder="Enter Prompt Here">{prompt}</textarea>
<hr>
{status}<br>
<hr>
<label>Gen. Amount</label> <input type="text" size="4" value="{max_length}" name="max_length"><br>
<label>Temperature</label> <input type="text" size="4" value="{temperature}" name="temperature"><br>
<label>Top-K</label> <input type="text" size="4" value="{top_k}" name="top_k"><br>
<label>Top-P</label> <input type="text" size="4" value="{top_p}" name="top_p"><br>
<label>Rep. Pen</label> <input type="text" size="4" value="{rep_pen}" name="rep_pen"><br>
<label>Ignore EOS</label> <input type="checkbox" name="use_default_badwordsids" value="1" {"checked" if use_default_badwordsids else ""}><br>
<input type="submit" name="generate" value="Generate"> (Please be patient)
</form>
<form action="/noscript">
<input type="submit" value="Reset">
</form>
</div>
</body></html>'''
        finalhtml = finalhtml.encode('utf-8')
        self.send_response(200)
        self.send_header('content-length', str(len(finalhtml)))
        self.end_headers(content_type='text/html')
        self.wfile.write(finalhtml)

    def do_GET(self):
        global maxctx, maxhordelen, friendlymodelname, KcppVersion, totalgens, preloaded_story, exitcounter
        self.path = self.path.rstrip('/')
        response_body = None
        content_type = 'application/json'

        if self.path in ["", "/?"] or self.path.startswith(('/?','?')): #it's possible for the root url to have ?params without /
            content_type = 'text/html'
            if self.embedded_kailite is None:
                response_body = (f"Embedded Kobold Lite is not found.<br>You will have to connect via the main KoboldAI client, or <a href='https://lite.koboldai.net?local=1&port={self.port}'>use this URL</a> to connect.").encode()
            else:
                response_body = self.embedded_kailite

        elif self.path in ["/noscript", "/noscript?"] or self.path.startswith(('/noscript?','noscript?')): #it's possible for the root url to have ?params without /
            self.noscript_webui()
            return

        elif self.path.endswith(('/api/v1/model', '/api/latest/model')):
            response_body = (json.dumps({'result': friendlymodelname }).encode())

        elif self.path.endswith(('/api/v1/config/max_length', '/api/latest/config/max_length')):
            response_body = (json.dumps({"value": maxhordelen}).encode())

        elif self.path.endswith(('/api/v1/config/max_context_length', '/api/latest/config/max_context_length')):
            response_body = (json.dumps({"value": min(maxctx,maxhordectx)}).encode())

        elif self.path.endswith(('/api/v1/config/soft_prompt', '/api/latest/config/soft_prompt')):
            response_body = (json.dumps({"value":""}).encode())

        elif self.path.endswith(('/api/v1/config/soft_prompts_list', '/api/latest/config/soft_prompts_list')):
            response_body = (json.dumps({"values": []}).encode())

        elif self.path.endswith(('/api/v1/info/version', '/api/latest/info/version')):
            response_body = (json.dumps({"result":"1.2.5"}).encode())

        elif self.path.endswith(('/api/extra/true_max_context_length')): #do not advertise this to horde
            response_body = (json.dumps({"value": maxctx}).encode())

        elif self.path.endswith(('/api/extra/version')):
            response_body = (json.dumps({"result":"KoboldCpp","version":KcppVersion}).encode())

        elif self.path.endswith(('/api/extra/perf')):
            lastp = handle.get_last_process_time()
            laste = handle.get_last_eval_time()
            lastc = handle.get_last_token_count()
            totalgens = handle.get_total_gens()
            stopreason = handle.get_last_stop_reason()
            response_body = (json.dumps({"last_process":lastp,"last_eval":laste,"last_token_count":lastc, "total_gens":totalgens, "stop_reason":stopreason, "queue":requestsinqueue, "idle":(0 if modelbusy.locked() else 1), "hordeexitcounter":exitcounter}).encode())

        elif self.path.endswith('/api/extra/generate/check'):
            pendtxtStr = ""
            if requestsinqueue==0 and totalgens>0:
                pendtxt = handle.get_pending_output()
                pendtxtStr = ctypes.string_at(pendtxt).decode("UTF-8","ignore")
            response_body = (json.dumps({"results": [{"text": pendtxtStr}]}).encode())

        elif self.path.endswith('/v1/models'):
            response_body = (json.dumps({"object":"list","data":[{"id":friendlymodelname,"object":"model","created":1,"owned_by":"koboldcpp","permission":[],"root":"koboldcpp"}]}).encode())

        elif self.path=="/api":
            content_type = 'text/html'
            if self.embedded_kcpp_docs is None:
                response_body = (f"KoboldCpp API is running!\n\nAPI usage reference can be found at the wiki: https://github.com/LostRuins/koboldcpp/wiki").encode()
            else:
                response_body = self.embedded_kcpp_docs

        elif self.path=="/v1":
            content_type = 'text/html'
            response_body = (f"KoboldCpp OpenAI compatible endpoint is running!\n\nFor usage reference, see https://platform.openai.com/docs/api-reference").encode()

        elif self.path=="/api/extra/preloadstory":
            if preloaded_story is None:
                response_body = (json.dumps({}).encode())
            else:
                response_body = preloaded_story
        elif self.path.endswith(('/api')) or self.path.endswith(('/api/v1')):
            self.path = "/api"
            self.send_response(302)
            self.send_header("location", self.path)
            self.end_headers(content_type='text/html')
            return None

        if response_body is None:
            self.send_response(404)
            self.end_headers(content_type='text/html')
            rp = 'Error: HTTP Server is running, but this endpoint does not exist. Please check the URL.'
            self.wfile.write(rp.encode())
        else:
            self.send_response(200)
            self.send_header('content-length', str(len(response_body)))
            self.end_headers(content_type=content_type)
            self.wfile.write(response_body)
        return

    def do_POST(self):
        global modelbusy, requestsinqueue, currentusergenkey, totalgens
        content_length = int(self.headers['content-length'])
        body = self.rfile.read(content_length)
        self.path = self.path.rstrip('/')
        response_body = None
        response_code = 200

        if self.path.endswith(('/api/extra/tokencount')):
            try:
                genparams = json.loads(body)
                countprompt = genparams.get('prompt', "")
                rawcountdata = handle.token_count(countprompt.encode("UTF-8"))
                countlimit = rawcountdata.count if (rawcountdata.count>=0 and rawcountdata.count<50000) else 0
                # the above protects the server in case the count limit got corrupted
                countdata = [rawcountdata.ids[i] for i in range(countlimit)]
                response_body = (json.dumps({"value": len(countdata),"ids": countdata}).encode())

            except Exception as e:
                utfprint("Count Tokens - Body Error: " + str(e))
                response_code = 400
                response_body = (json.dumps({"value": -1}).encode())

        elif self.path.endswith('/api/extra/abort'):
            multiuserkey = ""
            try:
                tempbody = json.loads(body)
                if isinstance(tempbody, dict):
                    multiuserkey = tempbody.get('genkey', "")
            except Exception as e:
                multiuserkey = ""
                pass

            if (multiuserkey=="" and requestsinqueue==0) or (multiuserkey!="" and multiuserkey==currentusergenkey):
                ag = handle.abort_generate()
                time.sleep(0.1) #short delay before replying
                response_body = (json.dumps({"success": ("true" if ag else "false")}).encode())
                print("\nGeneration Aborted")
            else:
                response_body = (json.dumps({"success": "false"}).encode())

        elif self.path.endswith('/api/extra/generate/check'):
            pendtxtStr = ""
            multiuserkey = ""
            try:
                tempbody = json.loads(body)
                if isinstance(tempbody, dict):
                    multiuserkey = tempbody.get('genkey', "")
            except Exception as e:
                multiuserkey = ""

            if totalgens>0:
                if (multiuserkey=="" and requestsinqueue==0) or (multiuserkey!="" and multiuserkey==currentusergenkey):
                    pendtxt = handle.get_pending_output()
                    pendtxtStr = ctypes.string_at(pendtxt).decode("UTF-8","ignore")
            response_body = (json.dumps({"results": [{"text": pendtxtStr}]}).encode())

        if response_body is not None:
            self.send_response(response_code)
            self.send_header('content-length', str(len(response_body)))
            self.end_headers(content_type='application/json')
            self.wfile.write(response_body)
            return

        reqblocking = False
        muint = int(args.multiuser)
        multiuserlimit = ((muint-1) if muint > 1 else 4)
        #backwards compatibility for up to 5 concurrent requests, use default limit of 5 if multiuser set to 1
        if muint > 0 and requestsinqueue < multiuserlimit:
            reqblocking = True
            requestsinqueue += 1
        if not modelbusy.acquire(blocking=reqblocking):
            self.send_response(503)
            self.end_headers(content_type='application/json')
            self.wfile.write(json.dumps({"detail": {
                    "msg": "Server is busy; please try again later.",
                    "type": "service_unavailable",
                }}).encode())
            return
        if reqblocking:
            requestsinqueue = (requestsinqueue - 1) if requestsinqueue > 0 else 0

        try:
            sse_stream_flag = False

            api_format = 0 #1=basic,2=kai,3=oai,4=oai-chat

            if self.path.endswith('/request'):
                api_format = 1

            if self.path.endswith(('/api/v1/generate', '/api/latest/generate')):
                api_format = 2

            if self.path.endswith('/api/extra/generate/stream'):
                api_format = 2
                sse_stream_flag = True

            if self.path.endswith('/v1/completions'):
                api_format = 3

            if self.path.endswith('/v1/chat/completions'):
                api_format = 4

            if api_format > 0:
                genparams = None
                try:
                    genparams = json.loads(body)
                except Exception as e:
                    utfprint("Body Err: " + str(body))
                    return self.send_response(503)

                is_quiet = args.quiet
                if (args.debugmode != -1 and not is_quiet) or args.debugmode >= 1:
                    utfprint("\nInput: " + json.dumps(genparams))

                if args.foreground:
                    bring_terminal_to_foreground()

                # Check if streaming chat completions, if so, set stream mode to true
                if (api_format == 4 or api_format == 3) and "stream" in genparams and genparams["stream"]:
                    sse_stream_flag = True

                gen = asyncio.run(self.handle_request(genparams, api_format, sse_stream_flag))

                try:
                    # Headers are already sent when streaming
                    if not sse_stream_flag:
                        self.send_response(200)
                        genresp = (json.dumps(gen).encode())
                        self.send_header('content-length', str(len(genresp)))
                        self.end_headers(content_type='application/json')
                        self.wfile.write(genresp)
                except:
                    print("Generate: The response could not be sent, maybe connection was terminated?")
                return
        finally:
            modelbusy.release()

        self.send_response(404)
        self.end_headers(content_type='text/html')


    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers(content_type='text/html')

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers(content_type='text/html')

    def end_headers(self, content_type=None):
        self.send_header('access-control-allow-origin', '*')
        self.send_header('access-control-allow-methods', '*')
        self.send_header('access-control-allow-headers', '*, Accept, Content-Type, Content-Length, Cache-Control, Accept-Encoding, X-CSRF-Token, Client-Agent, X-Fields, Content-Type, Authorization, X-Requested-With, X-HTTP-Method-Override, apikey, genkey')
        self.send_header("cache-control", "no-store")
        if content_type is not None:
            self.send_header('content-type', content_type)
        return super(ServerRequestHandler, self).end_headers()


def RunServerMultiThreaded(addr, port, embedded_kailite = None, embedded_kcpp_docs = None):
    global exitcounter
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((addr, port))
    sock.listen(5)

    class Thread(threading.Thread):
        def __init__(self, i):
            threading.Thread.__init__(self)
            self.i = i
            self.daemon = True
            self.start()

        def run(self):
            global exitcounter
            handler = ServerRequestHandler(addr, port, embedded_kailite, embedded_kcpp_docs)
            with http.server.HTTPServer((addr, port), handler, False) as self.httpd:
                try:
                    self.httpd.socket = sock
                    self.httpd.server_bind = self.server_close = lambda self: None
                    self.httpd.serve_forever()
                except (KeyboardInterrupt,SystemExit):
                    exitcounter = 999
                    self.httpd.server_close()
                    sys.exit(0)
                finally:
                    exitcounter = 999
                    self.httpd.server_close()
                    sys.exit(0)
        def stop(self):
            global exitcounter
            exitcounter = 999
            self.httpd.server_close()

    numThreads = 12
    threadArr = []
    for i in range(numThreads):
        threadArr.append(Thread(i))
    while 1:
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            global exitcounter
            exitcounter = 999
            for i in range(numThreads):
                threadArr[i].stop()
            sys.exit(0)

# note: customtkinter-5.2.0
def show_new_gui():
    from tkinter.filedialog import askopenfilename
    from tkinter.filedialog import asksaveasfile

    # if args received, launch
    if len(sys.argv) != 1:
        import tkinter as tk
        root = tk.Tk() #we dont want the useless window to be visible, but we want it in taskbar
        root.attributes("-alpha", 0)
        args.model_param = askopenfilename(title="Select ggml model .bin or .gguf file or .kcpps config")
        root.destroy()
        if args.model_param and args.model_param!="" and args.model_param.lower().endswith('.kcpps'):
            loadconfigfile(args.model_param)
        if not args.model_param:
            global exitcounter
            exitcounter = 999
            print("\nNo ggml model or kcpps file was selected. Exiting.")
            time.sleep(3)
            sys.exit(2)
        return

    import customtkinter as ctk
    nextstate = 0 #0=exit, 1=launch
    windowwidth = 540
    windowheight = 500
    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    root.geometry(str(windowwidth) + "x" + str(windowheight))
    root.title("KoboldCpp v"+KcppVersion)
    root.resizable(False,False)

    tabs = ctk.CTkFrame(root, corner_radius = 0, width=windowwidth, height=windowheight-50)
    tabs.grid(row=0, stick="nsew")
    tabnames= ["Quick Launch", "Hardware", "Tokens", "Model", "Network"]
    navbuttons = {}
    navbuttonframe = ctk.CTkFrame(tabs, width=100, height=int(tabs.cget("height")))
    navbuttonframe.grid(row=0, column=0, padx=2,pady=2)
    navbuttonframe.grid_propagate(False)

    tabcontentframe = ctk.CTkFrame(tabs, width=windowwidth - int(navbuttonframe.cget("width")), height=int(tabs.cget("height")))
    tabcontentframe.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
    tabcontentframe.grid_propagate(False)

    CLDevices = ["1","2","3","4"]
    CUDevices = ["1","2","3","4","All"]
    CLDevicesNames = ["","","",""]
    CUDevicesNames = ["","","","",""]
    MaxMemory = [0]

    tabcontent = {}
    lib_option_pairs = [
        (lib_openblas, "Use OpenBLAS"),
        (lib_clblast, "Use CLBlast"),
        (lib_cublas, "Use CuBLAS"),
        (lib_hipblas, "Use hipBLAS (ROCm)"),
        (lib_default, "Use No BLAS"),
        (lib_clblast_noavx2, "CLBlast NoAVX2 (Old CPU)"),
        (lib_noavx2, "NoAVX2 Mode (Old CPU)"),
        (lib_failsafe, "Failsafe Mode (Old CPU)")]
    openblas_option, clblast_option, cublas_option, hipblas_option, default_option, clblast_noavx2_option, noavx2_option, failsafe_option = (opt if file_exists(lib) or (os.name == 'nt' and file_exists(opt + ".dll")) else None for lib, opt in lib_option_pairs)
    # slider data
    blasbatchsize_values = ["-1", "32", "64", "128", "256", "512", "1024", "2048"]
    blasbatchsize_text = ["Don't Batch BLAS","32","64","128","256","512","1024","2048"]
    contextsize_text = ["256", "512", "1024", "2048", "3072", "4096", "6144", "8192", "12288", "16384", "24576", "32768", "65536"]
    runopts = [opt for lib, opt in lib_option_pairs if file_exists(lib)]
    antirunopts = [opt.replace("Use ", "") for lib, opt in lib_option_pairs if not (opt in runopts)]
    if os.name != 'nt':
        if "NoAVX2 Mode (Old CPU)" in antirunopts:
            antirunopts.remove("NoAVX2 Mode (Old CPU)")
        if "Failsafe Mode (Old CPU)" in antirunopts:
            antirunopts.remove("Failsafe Mode (Old CPU)")
        if "CLBlast NoAVX2 (Old CPU)" in antirunopts:
            antirunopts.remove("CLBlast NoAVX2 (Old CPU)")
    if not any(runopts):
        exitcounter = 999
        show_gui_msgbox("No Backends Available!","KoboldCPP couldn't locate any backends to use (i.e Default, OpenBLAS, CLBlast, CuBLAS).\n\nTo use the program, please run the 'make' command from the directory.")
        time.sleep(3)
        sys.exit(2)

    # Vars - should be in scope to be used by multiple widgets
    gpulayers_var = ctk.StringVar(value="0")
    threads_var = ctk.StringVar(value=str(default_threads))
    runopts_var = ctk.StringVar()
    gpu_choice_var = ctk.StringVar(value="1")

    launchbrowser = ctk.IntVar(value=1)
    highpriority = ctk.IntVar()
    disablemmap = ctk.IntVar()
    usemlock = ctk.IntVar()
    debugmode = ctk.IntVar()
    keepforeground = ctk.IntVar()
    quietmode = ctk.IntVar(value=0)

    lowvram_var = ctk.IntVar(value=1)
    mmq_var = ctk.IntVar(value=1)
    blas_threads_var = ctk.StringVar()
    blas_size_var = ctk.IntVar()
    version_var = ctk.StringVar(value="0")
    tensor_split_str_vars = ctk.StringVar(value="")

    contextshift = ctk.IntVar(value=1)
    remotetunnel = ctk.IntVar(value=0)
    smartcontext = ctk.IntVar()
    context_var = ctk.IntVar()
    customrope_var = ctk.IntVar()
    customrope_scale = ctk.StringVar(value="1.0")
    customrope_base = ctk.StringVar(value="10000")

    model_var = ctk.StringVar()
    lora_var = ctk.StringVar()
    lora_base_var = ctk.StringVar()
    preloadstory_var = ctk.StringVar()

    port_var = ctk.StringVar(value=defaultport)
    host_var = ctk.StringVar(value="")
    multiuser_var = ctk.IntVar(value=1)
    horde_name_var = ctk.StringVar(value="koboldcpp")
    horde_gen_var = ctk.StringVar(value=maxhordelen)
    horde_context_var = ctk.StringVar(value=maxhordectx)
    horde_apikey_var = ctk.StringVar(value="")
    horde_workername_var = ctk.StringVar(value="")
    usehorde_var = ctk.IntVar()

    def tabbuttonaction(name):
        for t in tabcontent:
            if name == t:
                tabcontent[t].grid(row=0, column=0)
                navbuttons[t].configure(fg_color="#6f727b")
            else:
                tabcontent[t].grid_forget()
                navbuttons[t].configure(fg_color="transparent")

    # Dynamically create tabs + buttons based on values of [tabnames]
    for idx, name in enumerate(tabnames):
        tabcontent[name] = ctk.CTkFrame(tabcontentframe, width=int(tabcontentframe.cget("width")), height=int(tabcontentframe.cget("height")), fg_color="transparent")
        tabcontent[name].grid_propagate(False)
        if idx == 0:
            tabcontent[name].grid(row=idx, sticky="nsew")
        ctk.CTkLabel(tabcontent[name], text= name, font=ctk.CTkFont(None, 14, 'bold')).grid(row=0, padx=12, pady = 5, stick='nw')

        navbuttons[name] = ctk.CTkButton(navbuttonframe, text=name, width = 100, corner_radius=0 , command = lambda d=name:tabbuttonaction(d), hover_color="#868a94" )
        navbuttons[name].grid(row=idx)

    tabbuttonaction(tabnames[0])
    # Quick Launch Tab
    quick_tab = tabcontent["Quick Launch"]

    # helper functions
    def makecheckbox(parent, text, variable=None, row=0, column=0, command=None, onvalue=1, offvalue=0):
        temp = ctk.CTkCheckBox(parent, text=text,variable=variable, onvalue=onvalue, offvalue=offvalue)
        if command is not None and variable is not None:
            variable.trace("w", command)
        temp.grid(row=row,column=column, padx=8, pady=1, stick="nw")
        return temp

    def makelabel(parent, text, row, column=0):
        temp = ctk.CTkLabel(parent, text=text)
        temp.grid(row=row, column=column, padx=8, pady=1, stick="nw")
        return temp

    def makeslider(parent, label, options, var, from_ , to,  row=0, width=160, height=10, set=0):
        sliderLabel = makelabel(parent, options[set], row + 1, 1)
        makelabel(parent, label, row)

        def sliderUpdate(a,b,c):
            sliderLabel.configure(text = options[int(var.get())])
        var.trace("w", sliderUpdate)
        slider = ctk.CTkSlider(parent, from_=from_, to=to, variable = var, width = width, height=height, border_width=5,number_of_steps=len(options) - 1)
        slider.grid(row=row+1,  column=0, padx = 8, stick="w")
        slider.set(set)
        return slider


    def makelabelentry(parent, text, var, row=0, width= 50):
        label = makelabel(parent, text, row)
        entry = ctk.CTkEntry(parent, width=width, textvariable=var) #you cannot set placeholder text for SHARED variables
        entry.grid(row=row, column=1, padx= 8, stick="nw")
        return entry, label


    def makefileentry(parent, text, searchtext, var, row=0, width=250, filetypes=[], onchoosefile=None):
        makelabel(parent, text, row)
        def getfilename(var, text):
            var.set(askopenfilename(title=text,filetypes=filetypes))
            if onchoosefile:
                onchoosefile(var.get())
        entry = ctk.CTkEntry(parent, width, textvariable=var)
        entry.grid(row=row+1, column=0, padx=8, stick="nw")
        button = ctk.CTkButton(parent, 50, text="Browse", command= lambda a=var,b=searchtext:getfilename(a,b))
        button.grid(row=row+1, column=1, stick="nw")
        return

    # decided to follow yellowrose's and kalomaze's suggestions, this function will automatically try to determine GPU identifiers
    # todo: autopick the right number of layers when a model is selected.
    # run in new thread so it doesnt block. does not return anything, instead overwrites specific values and redraws GUI
    def auto_gpu_heuristics():
        from subprocess import run, CalledProcessError
        FetchedCUdevices = []
        FetchedCUdeviceMem = []
        AMDgpu = None
        try: # Get OpenCL GPU names on windows using a special binary. overwrite at known index if found.
            basepath = os.path.abspath(os.path.dirname(__file__))
            output = run([((os.path.join(basepath, "winclinfo.exe")) if os.name == 'nt' else "clinfo"),"--json"], capture_output=True, text=True, check=True, encoding='utf-8').stdout
            data = json.loads(output)
            plat = 0
            dev = 0
            lowestclmem = 0
            for platform in data["devices"]:
                dev = 0
                for device in platform["online"]:
                    dname = device["CL_DEVICE_NAME"]
                    dmem = int(device["CL_DEVICE_GLOBAL_MEM_SIZE"])
                    idx = plat+dev*2
                    if idx<len(CLDevices):
                        CLDevicesNames[idx] = dname
                        lowestclmem = dmem if lowestclmem==0 else (dmem if dmem<lowestclmem else lowestclmem)
                    dev += 1
                plat += 1
            MaxMemory[0] = lowestclmem
        except Exception as e:
            pass

        try: # Get NVIDIA GPU names
            output = run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'], capture_output=True, text=True, check=True, encoding='utf-8').stdout
            FetchedCUdevices = [line.split(",")[0].strip() for line in output.splitlines()]
            FetchedCUdeviceMem = [line.split(",")[1].strip().split(" ")[0].strip() for line in output.splitlines()]
        except Exception as e:
            pass

        if len(FetchedCUdevices)==0:
            try: # Get AMD ROCm GPU names
                output = run(['rocminfo'], capture_output=True, text=True, check=True, encoding='utf-8').stdout
                device_name = None
                for line in output.splitlines(): # read through the output line by line
                    line = line.strip()
                    if line.startswith("Marketing Name:"): device_name = line.split(":", 1)[1].strip() # if we find a named device, temporarily save the name
                    elif line.startswith("Device Type:") and "GPU" in line and device_name is not None: # if the following Device Type is a GPU (not a CPU) then add it to devices list
                        FetchedCUdevices.append(device_name)
                        AMDgpu = True
                    elif line.startswith("Device Type:") and "GPU" not in line: device_name = None
                if FetchedCUdevices:
                    getamdvram = run(['rocm-smi', '--showmeminfo', 'vram', '--csv'], capture_output=True, text=True, check=True, encoding='utf-8').stdout # fetch VRAM of devices
                    FetchedCUdeviceMem = [line.split(",")[1].strip() for line in getamdvram.splitlines()[1:] if line.strip()]
            except Exception as e:
                pass

        for idx in range(0,4):
            if(len(FetchedCUdevices)>idx):
                CUDevicesNames[idx] = FetchedCUdevices[idx]
                if AMDgpu:
                    MaxMemory[0] = max(int(FetchedCUdeviceMem[idx]),MaxMemory[0])
                else:
                    MaxMemory[0] = max(int(FetchedCUdeviceMem[idx])*1024*1024,MaxMemory[0])

        #autopick cublas if suitable, requires at least 3.5GB VRAM to auto pick
        global exitcounter
        if exitcounter < 100 and MaxMemory[0]>3500000000 and CUDevicesNames[0]!="" and ("Use CuBLAS" in runopts or "Use hipBLAS (ROCM)" in runopts) and runopts_var.get()=="Use OpenBLAS":
            if "Use CuBLAS" in runopts:
                runopts_var.set("Use CuBLAS")
            elif "Use hipBLAS (ROCM)" in runopts:
                runopts_var.set("Use hipBLAS (ROCM)")

        changed_gpu_choice_var()
        return

    def on_picked_model_file(filepath):
        if filepath.lower().endswith('.kcpps'):
            #load it as a config file instead
            with open(filepath, 'r') as f:
                dict = json.load(f)
                import_vars(dict)
        else:
            autoset_gpu_layers(filepath)

    def autoset_gpu_layers(filepath): #shitty algo to determine how many layers to use
        try:
            global gui_layers_untouched
            fsize = os.path.getsize(filepath)
            if fsize>10000000: #dont bother with models < 10mb
                cs = int(contextsize_text[context_var.get()])
                mem = MaxMemory[0]
                layerlimit = 0

                if cs and cs > 4096:
                    fsize *= 1.2
                elif cs and cs > 2048:
                    fsize *= 1.1

                if mem < fsize*1.6:
                    sizeperlayer = fsize*0.052
                    layerlimit = int(min(200,mem/sizeperlayer))
                else:
                    layerlimit = 200 #assume full offload

                if layerlimit>=200:
                    lowvram_var.set(0)
                else:
                    lowvram_var.set(1)

                old_gui_layers_untouched = gui_layers_untouched
                gui_layers_zeroed = gpulayers_var.get()=="" or gpulayers_var.get()=="0"
                if (gui_layers_untouched or gui_layers_zeroed) and layerlimit>0:
                    gpulayers_var.set(str(layerlimit))
                    gui_layers_untouched = old_gui_layers_untouched
                    if gui_layers_zeroed:
                        gui_layers_untouched = True
        except Exception as ex:
            pass

    def show_tooltip(event, tooltip_text=None):
        if hasattr(show_tooltip, "_tooltip"):
            tooltip = show_tooltip._tooltip
        else:
            tooltip = ctk.CTkToplevel(root)
            tooltip.configure(fg_color="#ffffe0")
            tooltip.withdraw()
            tooltip.overrideredirect(True)
            tooltip_label = ctk.CTkLabel(tooltip, text=tooltip_text, text_color="#000000", fg_color="#ffffe0")
            tooltip_label.pack(expand=True, padx=2, pady=1)
            show_tooltip._tooltip = tooltip
        x, y = root.winfo_pointerxy()
        tooltip.wm_geometry(f"+{x + 10}+{y + 10}")
        tooltip.deiconify()

    def hide_tooltip(event):
        if hasattr(show_tooltip, "_tooltip"):
            tooltip = show_tooltip._tooltip
            tooltip.withdraw()

    def setup_backend_tooltip(parent):
        num_backends_built = makelabel(parent, str(len(runopts)) + f"/{7 if os.name == 'nt' else 4}", 5, 2)
        num_backends_built.grid(row=1, column=1, padx=195, pady=0)
        num_backends_built.configure(text_color="#00ff00")
        # Bind the backend count label with the tooltip function
        nl = '\n'
        num_backends_built.bind("<Enter>", lambda event: show_tooltip(event, f"Number of backends you have built and available." + (f"\n\nMissing Backends: \n\n{nl.join(antirunopts)}" if len(runopts) != 6 else "")))
        num_backends_built.bind("<Leave>", hide_tooltip)

    def changed_gpulayers(*args):
        global gui_layers_untouched
        gui_layers_untouched = False
        pass

    def changed_gpu_choice_var(*args):
        global exitcounter
        if exitcounter > 100:
            return
        if gpu_choice_var.get()!="All":
            try:
                s = int(gpu_choice_var.get())-1
                v = runopts_var.get()
                if v == "Use CLBlast" or v == "CLBlast NoAVX2 (Old CPU)":
                    quick_gpuname_label.configure(text=CLDevicesNames[s])
                    gpuname_label.configure(text=CLDevicesNames[s])
                else:
                    quick_gpuname_label.configure(text=CUDevicesNames[s])
                    gpuname_label.configure(text=CUDevicesNames[s])
            except Exception as ex:
                pass
        else:
            quick_gpuname_label.configure(text="")
            gpuname_label.configure(text="")

    gpu_choice_var.trace("w", changed_gpu_choice_var)
    gpulayers_var.trace("w", changed_gpulayers)

    def changerunmode(a,b,c):
        index = runopts_var.get()
        if index == "Use CLBlast" or index == "CLBlast NoAVX2 (Old CPU)" or index == "Use CuBLAS" or index == "Use hipBLAS (ROCm)":
            quick_gpuname_label.grid(row=3, column=1, padx=75, sticky="W")
            gpuname_label.grid(row=3, column=1, padx=75, sticky="W")
            gpu_selector_label.grid(row=3, column=0, padx = 8, pady=1, stick="nw")
            quick_gpu_selector_label.grid(row=3, column=0, padx = 8, pady=1, stick="nw")
            if index == "Use CLBlast" or index == "CLBlast NoAVX2 (Old CPU)":
                gpu_selector_box.grid(row=3, column=1, padx=8, pady=1, stick="nw")
                quick_gpu_selector_box.grid(row=3, column=1, padx=8, pady=1, stick="nw")
                if gpu_choice_var.get()=="All":
                    gpu_choice_var.set("1")
            elif index == "Use CuBLAS" or index == "Use hipBLAS (ROCm)":
                CUDA_gpu_selector_box.grid(row=3, column=1, padx=8, pady=1, stick="nw")
                CUDA_quick_gpu_selector_box.grid(row=3, column=1, padx=8, pady=1, stick="nw")
        else:
            quick_gpuname_label.grid_forget()
            gpuname_label.grid_forget()
            gpu_selector_label.grid_forget()
            gpu_selector_box.grid_forget()
            CUDA_gpu_selector_box.grid_forget()
            quick_gpu_selector_label.grid_forget()
            quick_gpu_selector_box.grid_forget()
            CUDA_quick_gpu_selector_box.grid_forget()

        if index == "Use CuBLAS" or index == "Use hipBLAS (ROCm)":
            lowvram_box.grid(row=4, column=0, padx=8, pady=1,  stick="nw")
            quick_lowvram_box.grid(row=4, column=0, padx=8, pady=1,  stick="nw")
            mmq_box.grid(row=4, column=1, padx=8, pady=1,  stick="nw")
            quick_mmq_box.grid(row=4, column=1, padx=8, pady=1,  stick="nw")
            tensor_split_label.grid(row=6, column=0, padx = 8, pady=1, stick="nw")
            tensor_split_entry.grid(row=6, column=1, padx=8, pady=1, stick="nw")
        else:
            lowvram_box.grid_forget()
            quick_lowvram_box.grid_forget()
            mmq_box.grid_forget()
            quick_mmq_box.grid_forget()
            tensor_split_label.grid_forget()
            tensor_split_entry.grid_forget()

        if index == "Use CLBlast" or index == "CLBlast NoAVX2 (Old CPU)" or index == "Use CuBLAS" or index == "Use hipBLAS (ROCm)":
            gpu_layers_label.grid(row=5, column=0, padx = 8, pady=1, stick="nw")
            gpu_layers_entry.grid(row=5, column=1, padx=8, pady=1, stick="nw")
            quick_gpu_layers_label.grid(row=5, column=0, padx = 8, pady=1, stick="nw")
            quick_gpu_layers_entry.grid(row=5, column=1, padx=8, pady=1, stick="nw")
        else:
            gpu_layers_label.grid_forget()
            gpu_layers_entry.grid_forget()
            quick_gpu_layers_label.grid_forget()
            quick_gpu_layers_entry.grid_forget()
        changed_gpu_choice_var()


    # presets selector
    makelabel(quick_tab, "Presets:", 1)

    runoptbox = ctk.CTkComboBox(quick_tab, values=runopts, width=180,variable=runopts_var, state="readonly")
    runoptbox.grid(row=1, column=1,padx=8, stick="nw")
    runoptbox.set(runopts[0]) # Set to first available option

    # Tell user how many backends are available
    setup_backend_tooltip(quick_tab)

    # gpu options
    quick_gpu_selector_label = makelabel(quick_tab, "GPU ID:", 3)
    quick_gpu_selector_box = ctk.CTkComboBox(quick_tab, values=CLDevices, width=60, variable=gpu_choice_var, state="readonly")
    CUDA_quick_gpu_selector_box = ctk.CTkComboBox(quick_tab, values=CUDevices, width=60, variable=gpu_choice_var, state="readonly")
    quick_gpuname_label = ctk.CTkLabel(quick_tab, text="")
    quick_gpuname_label.grid(row=3, column=1, padx=75, sticky="W")
    quick_gpuname_label.configure(text_color="#ffff00")
    quick_gpu_layers_entry,quick_gpu_layers_label = makelabelentry(quick_tab,"GPU Layers:", gpulayers_var, 5, 50)
    quick_lowvram_box = makecheckbox(quick_tab,  "Low VRAM", lowvram_var, 4,0)
    quick_mmq_box = makecheckbox(quick_tab,  "Use QuantMatMul (mmq)", mmq_var, 4,1)

    # threads
    makelabelentry(quick_tab, "Threads:" , threads_var, 8, 50)

    # blas batch size
    makeslider(quick_tab, "BLAS Batch Size:", blasbatchsize_text, blas_size_var, 0, 7, 12, set=5)

    # quick boxes
    quick_boxes = {"Launch Browser": launchbrowser , "High Priority" : highpriority, "Use SmartContext":smartcontext, "Disable MMAP":disablemmap,"Use ContextShift":contextshift,"Remote Tunnel":remotetunnel}
    for idx, name, in enumerate(quick_boxes):
        makecheckbox(quick_tab, name, quick_boxes[name], int(idx/2) +20, idx%2)
    # context size
    makeslider(quick_tab, "Context Size:", contextsize_text, context_var, 0, len(contextsize_text)-1, 30, set=3)

    # load model
    makefileentry(quick_tab, "Model:", "Select GGML Model File", model_var, 40, 170, onchoosefile=on_picked_model_file)

    # Hardware Tab
    hardware_tab = tabcontent["Hardware"]

    # presets selector
    makelabel(hardware_tab, "Presets:", 1)
    runoptbox = ctk.CTkComboBox(hardware_tab, values=runopts,  width=180,variable=runopts_var, state="readonly")
    runoptbox.grid(row=1, column=1,padx=8, stick="nw")
    runoptbox.set(runopts[0]) # Set to first available option

    # Tell user how many backends are available
    setup_backend_tooltip(hardware_tab)

    # gpu options
    gpu_selector_label = makelabel(hardware_tab, "GPU ID:", 3)
    gpu_selector_box = ctk.CTkComboBox(hardware_tab, values=CLDevices, width=60, variable=gpu_choice_var, state="readonly")
    CUDA_gpu_selector_box = ctk.CTkComboBox(hardware_tab, values=CUDevices, width=60, variable=gpu_choice_var, state="readonly")
    gpuname_label = ctk.CTkLabel(hardware_tab, text="")
    gpuname_label.grid(row=3, column=1, padx=75, sticky="W")
    gpuname_label.configure(text_color="#ffff00")
    gpu_layers_entry,gpu_layers_label = makelabelentry(hardware_tab,"GPU Layers:", gpulayers_var, 5, 50)
    tensor_split_entry,tensor_split_label = makelabelentry(hardware_tab, "Tensor Split:", tensor_split_str_vars, 6, 80)
    lowvram_box = makecheckbox(hardware_tab,  "Low VRAM", lowvram_var, 4,0)
    mmq_box = makecheckbox(hardware_tab,  "Use QuantMatMul (mmq)", mmq_var, 4,1)

    # threads
    makelabelentry(hardware_tab, "Threads:" , threads_var, 8, 50)

    # hardware checkboxes
    hardware_boxes = {"Launch Browser": launchbrowser , "High Priority" : highpriority, "Disable MMAP":disablemmap, "Use mlock":usemlock, "Debug Mode":debugmode, "Keep Foreground":keepforeground}

    for idx, name, in enumerate(hardware_boxes):
        makecheckbox(hardware_tab, name, hardware_boxes[name], int(idx/2) +30, idx%2)

    # blas thread specifier
    makelabelentry(hardware_tab, "BLAS threads:" , blas_threads_var, 11, 50)
    # blas batch size
    makeslider(hardware_tab, "BLAS Batch Size:", blasbatchsize_text, blas_size_var, 0, 7, 12, set=5)
    # force version
    makelabelentry(hardware_tab, "Force Version:" , version_var, 100, 50)

    runopts_var.trace('w', changerunmode)
    changerunmode(1,1,1)

    # Tokens Tab
    tokens_tab = tabcontent["Tokens"]
    # tokens checkboxes
    token_boxes = {"Use SmartContext":smartcontext, "Use ContextShift":contextshift}
    for idx, name, in enumerate(token_boxes):
        makecheckbox(tokens_tab, name, token_boxes[name], idx + 1)

    # context size
    makeslider(tokens_tab, "Context Size:",contextsize_text, context_var, 0, len(contextsize_text)-1, 20, set=3)


    customrope_scale_entry, customrope_scale_label = makelabelentry(tokens_tab, "RoPE Scale:", customrope_scale)
    customrope_base_entry, customrope_base_label = makelabelentry(tokens_tab, "RoPE Base:", customrope_base)
    def togglerope(a,b,c):
        items = [customrope_scale_label, customrope_scale_entry,customrope_base_label, customrope_base_entry]
        for idx, item in enumerate(items):
            if customrope_var.get() == 1:
                item.grid(row=23 + int(idx/2), column=idx%2, padx=8, stick="nw")
            else:
                item.grid_forget()
    makecheckbox(tokens_tab,  "Custom RoPE Config", variable=customrope_var, row=22, command=togglerope)
    togglerope(1,1,1)

    # Model Tab
    model_tab = tabcontent["Model"]

    makefileentry(model_tab, "Model:", "Select GGML Model File", model_var, 1, onchoosefile=on_picked_model_file)
    makefileentry(model_tab, "Lora:", "Select Lora File",lora_var, 3)
    makefileentry(model_tab, "Lora Base:", "Select Lora Base File", lora_base_var, 5)
    makefileentry(model_tab, "Preloaded Story:", "Select Preloaded Story File", preloadstory_var, 7)

    # Network Tab
    network_tab = tabcontent["Network"]

    # interfaces
    makelabelentry(network_tab, "Port: ", port_var, 1, 150)
    makelabelentry(network_tab, "Host: ", host_var, 2, 150)

    makecheckbox(network_tab, "Multiuser Mode", multiuser_var, 3)
    makecheckbox(network_tab, "Remote Tunnel", remotetunnel, 3, 1)
    makecheckbox(network_tab, "Quiet Mode", quietmode, 4)

    # horde
    makelabel(network_tab, "Horde:", 5).grid(pady=10)

    horde_name_entry,  horde_name_label = makelabelentry(network_tab, "Horde Model Name:", horde_name_var, 10, 180)
    horde_gen_entry,  horde_gen_label = makelabelentry(network_tab, "Gen. Length:", horde_gen_var, 11, 50)
    horde_context_entry,  horde_context_label = makelabelentry(network_tab, "Max Context:",horde_context_var, 12, 50)
    horde_apikey_entry,  horde_apikey_label = makelabelentry(network_tab, "API Key (If Embedded Worker):",horde_apikey_var, 13, 180)
    horde_workername_entry,  horde_workername_label = makelabelentry(network_tab, "Horde Worker Name:",horde_workername_var, 14, 180)

    def togglehorde(a,b,c):
        labels = [horde_name_label, horde_gen_label, horde_context_label, horde_apikey_label, horde_workername_label]
        for idx, item in enumerate([horde_name_entry, horde_gen_entry, horde_context_entry, horde_apikey_entry, horde_workername_entry]):
            if usehorde_var.get() == 1:
                item.grid(row=10 + idx, column = 1, padx=8, pady=1, stick="nw")
                labels[idx].grid(row=10 + idx, padx=8, pady=1, stick="nw")
            else:
                item.grid_forget()
                labels[idx].grid_forget()
        if usehorde_var.get()==1 and (horde_name_var.get()=="koboldcpp" or horde_name_var.get()=="") and model_var.get()!="":
            basefile = os.path.basename(model_var.get())
            horde_name_var.set(sanitize_string(os.path.splitext(basefile)[0]))

    makecheckbox(network_tab, "Configure for Horde", usehorde_var, 6, command=togglehorde)
    togglehorde(1,1,1)

    # launch
    def guilaunch():
        if model_var.get() == "":
            tmp = askopenfilename(title="Select ggml model .bin or .gguf file")
            model_var.set(tmp)
        nonlocal nextstate
        nextstate = 1
        root.destroy()
        pass

    def export_vars():
        args.threads = int(threads_var.get())
        args.usemlock   = usemlock.get() == 1
        args.debugmode  = debugmode.get()
        args.launch     = launchbrowser.get()==1
        args.highpriority = highpriority.get()==1
        args.nommap = disablemmap.get()==1
        args.smartcontext = smartcontext.get()==1
        args.noshift = contextshift.get()==0
        args.remotetunnel = remotetunnel.get()==1
        args.foreground = keepforeground.get()==1
        args.quiet = quietmode.get()==1

        gpuchoiceidx = 0
        if gpu_choice_var.get()!="All":
            gpuchoiceidx = int(gpu_choice_var.get())-1
        if runopts_var.get() == "Use CLBlast" or runopts_var.get() == "CLBlast NoAVX2 (Old CPU)":
            args.useclblast = [[0,0], [1,0], [0,1], [1,1]][gpuchoiceidx]
            if runopts_var.get() == "CLBlast NoAVX2 (Old CPU)":
                args.noavx2 = True
        if runopts_var.get() == "Use CuBLAS" or runopts_var.get() == "Use hipBLAS (ROCm)":
            if gpu_choice_var.get()=="All":
                args.usecublas = ["lowvram"] if lowvram_var.get() == 1 else ["normal"]
            else:
                args.usecublas = ["lowvram",str(gpuchoiceidx)] if lowvram_var.get() == 1 else ["normal",str(gpuchoiceidx)]
            if mmq_var.get()==1:
                args.usecublas.append("mmq")
        if gpulayers_var.get():
            args.gpulayers = int(gpulayers_var.get())
        if runopts_var.get()=="Use No BLAS":
            args.noblas = True
        if runopts_var.get()=="NoAVX2 Mode (Old CPU)":
            args.noavx2 = True
        if runopts_var.get()=="Failsafe Mode (Old CPU)":
            args.noavx2 = True
            args.noblas = True
            args.nommap = True
        if tensor_split_str_vars.get()!="":
            tssv = tensor_split_str_vars.get()
            if "," in tssv:
                args.tensor_split = [float(x) for x in tssv.split(",")]
            else:
                args.tensor_split = [float(x) for x in tssv.split(" ")]

        args.blasthreads = None if blas_threads_var.get()=="" else int(blas_threads_var.get())

        args.blasbatchsize = int(blasbatchsize_values[int(blas_size_var.get())])
        args.forceversion = 0 if version_var.get()=="" else int(version_var.get())

        args.contextsize = int(contextsize_text[context_var.get()])

        if customrope_var.get()==1:
            args.ropeconfig = [float(customrope_scale.get()),float(customrope_base.get())]

        args.model_param = None if model_var.get() == "" else model_var.get()
        args.lora = None if lora_var.get() == "" else ([lora_var.get()] if lora_base_var.get()=="" else [lora_var.get(), lora_base_var.get()])
        args.preloadstory = None if preloadstory_var.get() == "" else preloadstory_var.get()

        args.port_param = defaultport if port_var.get()=="" else int(port_var.get())
        args.host = host_var.get()
        args.multiuser = multiuser_var.get()

        if horde_apikey_var.get()=="" or horde_workername_var.get()=="":
            args.hordeconfig = None if usehorde_var.get() == 0 else [horde_name_var.get(), horde_gen_var.get(), horde_context_var.get()]
        else:
            args.hordeconfig = None if usehorde_var.get() == 0 else [horde_name_var.get(), horde_gen_var.get(), horde_context_var.get(), horde_apikey_var.get(), horde_workername_var.get()]

    def import_vars(dict):
        if "threads" in dict:
            threads_var.set(dict["threads"])
        usemlock.set(1 if "usemlock" in dict and dict["usemlock"] else 0)
        if "debugmode" in dict:
            debugmode.set(dict["debugmode"])
        launchbrowser.set(1 if "launch" in dict and dict["launch"] else 0)
        highpriority.set(1 if "highpriority" in dict and dict["highpriority"] else 0)
        disablemmap.set(1 if "nommap" in dict and dict["nommap"] else 0)
        smartcontext.set(1 if "smartcontext" in dict and dict["smartcontext"] else 0)
        contextshift.set(0 if "noshift" in dict and dict["noshift"] else 1)
        remotetunnel.set(1 if "remotetunnel" in dict and dict["remotetunnel"] else 0)
        keepforeground.set(1 if "foreground" in dict and dict["foreground"] else 0)
        quietmode.set(1 if "quiet" in dict and dict["quiet"] else 0)
        if "useclblast" in dict and dict["useclblast"]:
            if "noavx2" in dict and dict["noavx2"]:
                if clblast_noavx2_option is not None:
                    runopts_var.set(clblast_noavx2_option)
                    gpu_choice_var.set(str(["0 0", "1 0", "0 1", "1 1"].index(str(dict["useclblast"][0]) + " " + str(dict["useclblast"][1])) + 1))
            else:
                if clblast_option is not None:
                    runopts_var.set(clblast_option)
                    gpu_choice_var.set(str(["0 0", "1 0", "0 1", "1 1"].index(str(dict["useclblast"][0]) + " " + str(dict["useclblast"][1])) + 1))
        elif "usecublas" in dict and dict["usecublas"]:
            if cublas_option is not None or hipblas_option is not None:
                if cublas_option:
                    runopts_var.set(cublas_option)
                elif hipblas_option:
                    runopts_var.set(cublas_option)
                lowvram_var.set(1 if "lowvram" in dict["usecublas"] else 0)
                mmq_var.set(1 if "mmq" in dict["usecublas"] else 0)
                gpu_choice_var.set("All")
                for g in range(4):
                    if str(g) in dict["usecublas"]:
                        gpu_choice_var.set(str(g+1))
                        break
        elif  "noavx2" in dict and "noblas" in dict and dict["noblas"] and dict["noavx2"]:
            if failsafe_option is not None:
                runopts_var.set(failsafe_option)
        elif "noavx2" in dict and dict["noavx2"]:
            if noavx2_option is not None:
                runopts_var.set(noavx2_option)
        elif "noblas" in dict and dict["noblas"]:
            if default_option is not None:
                runopts_var.set(default_option)
        elif openblas_option is not None:
                runopts_var.set(openblas_option)
        if "gpulayers" in dict and dict["gpulayers"]:
            gpulayers_var.set(dict["gpulayers"])
        if "tensor_split" in dict and dict["tensor_split"]:
            tssep = ','.join(map(str, dict["tensor_split"]))
            tensor_split_str_vars.set(tssep)
        if "blasthreads" in dict and dict["blasthreads"]:
            blas_threads_var.set(str(dict["blasthreads"]))
        else:
            blas_threads_var.set("")
        if "contextsize" in dict and dict["contextsize"]:
            context_var.set(contextsize_text.index(str(dict["contextsize"])))
        if "ropeconfig" in dict and dict["ropeconfig"] and len(dict["ropeconfig"])>1:
            if dict["ropeconfig"][0]>0:
                customrope_var.set(1)
                customrope_scale.set(str(dict["ropeconfig"][0]))
                customrope_base.set(str(dict["ropeconfig"][1]))
            else:
                customrope_var.set(0)

        if "blasbatchsize" in dict and dict["blasbatchsize"]:
            blas_size_var.set(blasbatchsize_values.index(str(dict["blasbatchsize"])))
        if "forceversion" in dict and dict["forceversion"]:
            version_var.set(str(dict["forceversion"]))

        if "model_param" in dict and dict["model_param"]:
            model_var.set(dict["model_param"])

        if "lora" in dict and dict["lora"]:
            if len(dict["lora"]) > 1:
                lora_var.set(dict["lora"][0])
                lora_base_var.set(dict["lora"][1])
            else:
                lora_var.set(dict["lora"][0])

        if "preloadstory" in dict and dict["preloadstory"]:
            preloadstory_var.set(dict["preloadstory"])

        if "port_param" in dict and dict["port_param"]:
            port_var.set(dict["port_param"])

        if "host" in dict and dict["host"]:
            host_var.set(dict["host"])

        if "multiuser" in dict:
            multiuser_var.set(dict["multiuser"])

        if "hordeconfig" in dict and dict["hordeconfig"] and len(dict["hordeconfig"]) > 1:
            horde_name_var.set(dict["hordeconfig"][0])
            horde_gen_var.set(dict["hordeconfig"][1])
            horde_context_var.set(dict["hordeconfig"][2])
            if len(dict["hordeconfig"]) > 4:
                horde_apikey_var.set(dict["hordeconfig"][3])
                horde_workername_var.set(dict["hordeconfig"][4])
                usehorde_var.set("1")

    def save_config():
        file_type = [("KoboldCpp Settings", "*.kcpps")]
        filename = asksaveasfile(filetypes=file_type, defaultextension=file_type)
        if filename == None: return
        export_vars()
        file = open(str(filename.name), 'a')
        file.write(json.dumps(args.__dict__))
        file.close()
        pass

    def load_config():
        file_type = [("KoboldCpp Settings", "*.kcpps")]
        filename = askopenfilename(filetypes=file_type, defaultextension=file_type)
        if not filename or filename=="":
            return
        with open(filename, 'r') as f:
            dict = json.load(f)
            import_vars(dict)
        pass

    def display_help():
        try:
            import webbrowser as wb
            wb.open("https://github.com/LostRuins/koboldcpp/wiki")
        except:
            print("Cannot launch help in browser.")
    def display_updates():
        try:
            import webbrowser as wb
            wb.open("https://github.com/LostRuins/koboldcpp/releases/latest")
        except:
            print("Cannot launch updates in browser.")

    ctk.CTkButton(tabs , text = "Launch", fg_color="#2f8d3c", hover_color="#2faa3c", command = guilaunch, width=80, height = 35 ).grid(row=1,column=1, stick="se", padx= 25, pady=5)

    ctk.CTkButton(tabs , text = "Update", fg_color="#9900cc", hover_color="#aa11dd", command = display_updates, width=90, height = 35 ).grid(row=1,column=0, stick="sw", padx= 5, pady=5)
    ctk.CTkButton(tabs , text = "Save", fg_color="#084a66", hover_color="#085a88", command = save_config, width=60, height = 35 ).grid(row=1,column=1, stick="sw", padx= 5, pady=5)
    ctk.CTkButton(tabs , text = "Load", fg_color="#084a66", hover_color="#085a88", command = load_config, width=60, height = 35 ).grid(row=1,column=1, stick="sw", padx= 70, pady=5)
    ctk.CTkButton(tabs , text = "Help", fg_color="#992222", hover_color="#bb3333", command = display_help, width=60, height = 35 ).grid(row=1,column=1, stick="sw", padx= 135, pady=5)

    # start a thread that tries to get actual gpu names and layer counts
    gpuinfo_thread = threading.Thread(target=auto_gpu_heuristics)
    gpuinfo_thread.start() #submit job in new thread so nothing is waiting

    # runs main loop until closed or launch clicked
    root.mainloop()

    if nextstate==0:
        exitcounter = 999
        print("Exiting by user request.")
        time.sleep(3)
        sys.exit(0)
    else:
        # processing vars
        export_vars()

        if not args.model_param:
            exitcounter = 999
            print("\nNo ggml model file was selected. Exiting.")
            time.sleep(3)
            sys.exit(2)

def show_gui_msgbox(title,message):
    print(title + ": " + message)
    try:
        from tkinter import messagebox
        import tkinter as tk
        root = tk.Tk()
        root.attributes("-alpha", 0)
        messagebox.showerror(title=title, message=message)
        root.destroy()
    except Exception as ex2:
        pass

def print_with_time(txt):
    from datetime import datetime
    print(f"{datetime.now().strftime('[%H:%M:%S]')} " + txt)

def make_url_request(url, data, method='POST', headers={}):
    import urllib.request
    try:
        request = None
        if method=='POST':
            json_payload = json.dumps(data).encode('utf-8')
            request = urllib.request.Request(url, data=json_payload, headers=headers, method=method)
            request.add_header('content-type', 'application/json')
        else:
            request = urllib.request.Request(url, headers=headers, method=method)
        response_data = ""
        with urllib.request.urlopen(request) as response:
            response_data = response.read().decode('utf-8')
            json_response = json.loads(response_data)
            return json_response
    except urllib.error.HTTPError as e:
        try:
            errmsg = e.read().decode('utf-8')
            print_with_time(f"Error: {e} - {errmsg}")
        except Exception as e:
            print_with_time(f"Error: {e}")
        return None
    except Exception as e:
        print_with_time(f"Error: {e} - {response_data}")
        return None

#A very simple and stripped down embedded horde worker with no dependencies
def run_horde_worker(args, api_key, worker_name):
    from datetime import datetime
    global friendlymodelname, maxhordectx, maxhordelen, exitcounter, punishcounter, modelbusy, session_starttime
    epurl = f"http://localhost:{args.port}"
    if args.host!="":
        epurl = f"http://{args.host}:{args.port}"

    def submit_completed_generation(url, jobid, sessionstart, submit_dict):
        global exitcounter, punishcounter, session_kudos_earned, session_jobs, rewardcounter
        reply = make_url_request_horde(url, submit_dict)
        if not reply:
            punishcounter += 1
            print_with_time(f"Error, Job submit failed.")
        else:
            reward = reply["reward"]
            session_kudos_earned += reward
            session_jobs += 1
            curtime = datetime.now()
            elapsedtime=curtime-sessionstart
            hrs = int(elapsedtime.total_seconds()) // 3600
            mins = elapsedtime.seconds // 60 % 60
            secs = elapsedtime.seconds % 60
            elapsedtimestr = f"{hrs:03d}h:{mins:02d}m:{secs:02d}s"
            earnrate = session_kudos_earned/(elapsedtime.total_seconds()/3600)
            print_with_time(f'Submitted {jobid} and earned {reward:.0f} kudos\n[Total:{session_kudos_earned:.0f} kudos, Time:{elapsedtimestr}, Jobs:{session_jobs}, EarnRate:{earnrate:.0f} kudos/hr]')
            rewardcounter += 1
            if rewardcounter > 50:
                rewardcounter = 0
                if exitcounter > 1:
                    exitcounter -= 1

    def make_url_request_horde(url, data, method='POST'):
        headers = headers = {"apikey": api_key,'User-Agent':'KoboldCppEmbeddedWorkerV2','Client-Agent':'KoboldCppEmbedWorker:2'}
        ret = make_url_request(url, data, method, headers)
        if not ret:
            print("Make sure your Horde API key and worker name is valid!")
        return ret

    current_id = None
    current_payload = None
    current_generation = None
    session_starttime = datetime.now()
    sleepy_counter = 0 #if this exceeds a value, worker becomes sleepy (slower)
    exitcounter = 0
    print(f"===\nEmbedded Horde Worker '{worker_name}' Starting...\n(To use your own KAI Bridge/Scribe worker instead, don't set your API key)")
    BRIDGE_AGENT = f"KoboldCppEmbedWorker:2:https://github.com/LostRuins/koboldcpp"
    cluster = "https://horde.koboldai.net"
    while exitcounter < 10:
        time.sleep(3)
        readygo = make_url_request_horde(f'{epurl}/api/v1/info/version', None,'GET')
        if readygo:
            print_with_time(f"Embedded Horde Worker '{worker_name}' is started.")
            break

    while exitcounter < 10:
        currentjob_attempts = 0
        current_generation = None

        if punishcounter >= 5:
            punishcounter = 0
            exitcounter += 1
            if exitcounter < 10:
                penaltytime = (2 ** exitcounter)
                print_with_time(f"Horde Worker Paused for {penaltytime} min - Too many errors. It will resume automatically, but you should restart it.")
                print_with_time(f"Caution: Too many failed jobs may lead to entering maintenance mode.")
                time.sleep(60 * penaltytime)
            else:
                 print_with_time(f"Horde Worker Exit limit reached, too many errors.")

        #first, make sure we are not generating
        if modelbusy.locked():
            time.sleep(0.2)
            continue

        #pop new request
        gen_dict = {
            "name": worker_name,
            "models": [friendlymodelname],
            "max_length": maxhordelen,
            "max_context_length": maxhordectx,
            "priority_usernames": [],
            "softprompts": [],
            "bridge_agent": BRIDGE_AGENT,
        }
        pop = make_url_request_horde(f'{cluster}/api/v2/generate/text/pop',gen_dict)
        if not pop:
            punishcounter += 1
            print_with_time(f"Failed to fetch job from {cluster}. Waiting 10 seconds...")
            time.sleep(10)
            continue
        if not pop["id"]:
            slp = (1 if sleepy_counter<10 else (2 if sleepy_counter<25 else 3))
            time.sleep(slp)
            sleepy_counter += 1
            if sleepy_counter==20:
                print_with_time(f"No recent jobs, entering low power mode...")
            continue

        sleepy_counter = 0
        current_id = pop['id']
        current_payload = pop['payload']
        print(f"") #empty newline
        print_with_time(f"Job received from {cluster} for {current_payload.get('max_length',80)} tokens and {current_payload.get('max_context_length',1024)} max context. Starting generation...")

        #do gen
        while exitcounter < 10:
            if not modelbusy.locked():
                current_generation = make_url_request_horde(f'{epurl}/api/v1/generate', current_payload)
                if current_generation:
                    break
                else:
                    currentjob_attempts += 1
                    if currentjob_attempts>5:
                        break
            print_with_time(f"Server Busy - Not ready to generate...")
            time.sleep(5)

        #submit reply
        print(f"") #empty newline
        if current_generation:
            submit_dict = {
                "id": current_id,
                "generation": current_generation["results"][0]["text"],
                "state": "ok"
            }
            submiturl = cluster + '/api/v2/generate/text/submit'
            submit_thread = threading.Thread(target=submit_completed_generation, args=(submiturl, current_id, session_starttime, submit_dict))
            submit_thread.start() #submit job in new thread so nothing is waiting
        else:
            print_with_time(f"Error, Abandoned current job due to errors. Getting new job.")
        current_id = None
        current_payload = None
        time.sleep(0.1)

    if exitcounter<100:
        print_with_time(f"Horde Worker Shutdown - Too many errors.")
    else:
        print_with_time(f"Horde Worker Shutdown - Server Closing.")
    exitcounter = 999
    time.sleep(3)
    sys.exit(2)

def setuptunnel():
    # This script will help setup a cloudflared tunnel for accessing KoboldCpp over the internet
    # It should work out of the box on both linux and windows
    try:
        import subprocess, re

        def run_tunnel():
            tunnelproc = None
            tunneloutput = ""
            tunnelrawlog = ""
            time.sleep(0.2)
            if os.name == 'nt':
                print("Starting Cloudflare Tunnel for Windows, please wait...")
                tunnelproc = subprocess.Popen(f"cloudflared.exe tunnel --url localhost:{args.port}", text=True, encoding='utf-8', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            else:
                print("Starting Cloudflare Tunnel for Linux, please wait...")
                tunnelproc = subprocess.Popen(f"./cloudflared-linux-amd64 tunnel --url http://localhost:{args.port}", text=True, encoding='utf-8', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            time.sleep(10)
            def tunnel_reader():
                nonlocal tunnelproc,tunneloutput,tunnelrawlog
                pattern = r'https://[\w\.-]+\.trycloudflare\.com'
                while True:
                    line = tunnelproc.stderr.readline() #cloudflare writes to stderr for some reason
                    tunnelrawlog += line+"\n"
                    if not line:
                        return
                    found = re.findall(pattern, line)
                    for x in found:
                        tunneloutput = x

                        print(f"Your remote Kobold API can be found at {tunneloutput}/api")
                        print(f"Your remote OpenAI Compatible API can be found at {tunneloutput}/v1")
                        print("======\n")
                        print(f"Your remote tunnel is ready, please connect to {tunneloutput}")
                        return

            tunnel_reader_thread = threading.Thread(target=tunnel_reader)
            tunnel_reader_thread.start()
            time.sleep(5)
            if tunneloutput=="":
                print(f"Error: Could not create cloudflare tunnel!\nMore Info:\n{tunnelrawlog}")
            time.sleep(0.5)
            tunnelproc.wait()

        if os.name == 'nt':
            print("Downloading Cloudflare Tunnel for Windows...")
            subprocess.run("curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe -o cloudflared.exe", shell=True, capture_output=True, text=True, check=True, encoding='utf-8')
        else:
            print("Downloading Cloudflare Tunnel for Linux...")
            subprocess.run("curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared-linux-amd64", shell=True, capture_output=True, text=True, check=True, encoding='utf-8')
            subprocess.run("chmod +x 'cloudflared-linux-amd64'", shell=True)
        tunnel_thread = threading.Thread(target=run_tunnel)
        tunnel_thread.start()
    except Exception as ex:
        print("Remote Tunnel Failed!")
        print(str(ex))
        return None

def unload_libs():
    global handle
    import platform
    OS = platform.system()
    dll_close = None
    if OS == "Windows":  # pragma: Windows
        from ctypes import wintypes
        dll_close = ctypes.windll.kernel32.FreeLibrary
        dll_close.argtypes = [wintypes.HMODULE]
        dll_close.restype = ctypes.c_int
    elif OS == "Darwin":
        try:
            try:  # macOS 11 (Big Sur). Possibly also later macOS 10s.
                stdlib = ctypes.CDLL("libc.dylib")
            except OSError:
                stdlib = ctypes.CDLL("libSystem")
        except OSError:
            # Older macOSs. Not only is the name inconsistent but it's
            # not even in PATH.
            stdlib = ctypes.CDLL("/usr/lib/system/libsystem_c.dylib")
        dll_close = stdlib.dlclose
        dll_close.argtypes = [ctypes.c_void_p]
        dll_close.restype = ctypes.c_int
    elif OS == "Linux":
        try:
            stdlib = ctypes.CDLL("")
        except OSError:
            stdlib = ctypes.CDLL("libc.so") # Alpine Linux.
        dll_close = stdlib.dlclose
        dll_close.argtypes = [ctypes.c_void_p]
        dll_close.restype = ctypes.c_int
    elif sys.platform == "msys":
        # msys can also use `ctypes.CDLL("kernel32.dll").FreeLibrary()`.
        stdlib = ctypes.CDLL("msys-2.0.dll")
        dll_close = stdlib.dlclose
        dll_close.argtypes = [ctypes.c_void_p]
        dll_close.restype = ctypes.c_int
    elif sys.platform == "cygwin":
        stdlib = ctypes.CDLL("cygwin1.dll")
        dll_close = stdlib.dlclose
        dll_close.argtypes = [ctypes.c_void_p]
        dll_close.restype = ctypes.c_int
    elif OS == "FreeBSD":
        # FreeBSD uses `/usr/lib/libc.so.7` where `7` is another version number.
        # It is not in PATH but using its name instead of its path is somehow the
        # only way to open it. The name must include the .so.7 suffix.
        stdlib = ctypes.CDLL("libc.so.7")
        dll_close = stdlib.close

    if handle and dll_close:
        print("Unloading Libraries...")
        dll_close(handle._handle)
        del handle.load_model
        del handle.generate
        del handle.new_token
        del handle.get_stream_count
        del handle.has_finished
        del handle.get_last_eval_time
        del handle.get_last_process_time
        del handle.get_last_token_count
        del handle.get_total_gens
        del handle.get_last_stop_reason
        del handle.abort_generate
        del handle.token_count
        del handle.get_pending_output
        del handle
        handle = None

def loadconfigfile(filename):
    print("Loading kcpps configuration file...")
    with open(filename, 'r') as f:
        config = json.load(f)
        for key, value in config.items():
            setattr(args, key, value)

def sanitize_string(input_string):
    # alphanumeric characters, dots, dashes, and underscores
    import re
    sanitized_string = re.sub( r'[^\w\d\.\-_]', '', input_string)
    return sanitized_string

def main(launch_args,start_server=True):
    global args, friendlymodelname
    args = launch_args
    embedded_kailite = None
    embedded_kcpp_docs = None
    if args.config and len(args.config)==1:
        if isinstance(args.config[0], str) and os.path.exists(args.config[0]):
           loadconfigfile(args.config[0])
        else:
            global exitcounter
            exitcounter = 999
            print("Specified kcpp config file invalid or not found.")
            time.sleep(3)
            sys.exit(2)

    #positional handling for kcpps files (drag and drop)
    if args.model_param and args.model_param!="" and args.model_param.lower().endswith('.kcpps'):
        loadconfigfile(args.model_param)

    if not args.model_param:
        args.model_param = args.model

    if not args.model_param:
        #give them a chance to pick a file
        print("For command line arguments, please refer to --help")
        print("***")
        try:
            show_new_gui()
        except Exception as ex:
            exitcounter = 999
            ermsg = "Reason: " + str(ex) + "\nFile selection GUI unsupported.\ncustomtkinter python module required!\nPlease check command line: script.py --help"
            show_gui_msgbox("Warning, GUI failed to start",ermsg)
            time.sleep(3)
            sys.exit(2)

    #try to read story if provided
    if args.preloadstory:
        if isinstance(args.preloadstory, str) and os.path.exists(args.preloadstory):
            print(f"Preloading saved story {args.preloadstory} into server...")
            with open(args.preloadstory, mode='rb') as f:
                global preloaded_story
                preloaded_story = f.read()
                print("Saved story preloaded.")
        else:
            print(f"Warning: Saved story file {args.preloadstory} invalid or not found. No story will be preloaded into server.")

    # sanitize and replace the default vanity name. remember me....
    if args.model_param!="":
        newmdldisplayname = os.path.basename(args.model_param)
        newmdldisplayname = os.path.splitext(newmdldisplayname)[0]
        friendlymodelname = "koboldcpp/" + sanitize_string(newmdldisplayname)

    if args.hordeconfig and args.hordeconfig[0]!="":
        global maxhordelen, maxhordectx, showdebug
        friendlymodelname = args.hordeconfig[0]
        if args.debugmode == 1:
            friendlymodelname = "debug-" + friendlymodelname
        if not friendlymodelname.startswith("koboldcpp/"):
            friendlymodelname = "koboldcpp/" + friendlymodelname
        if len(args.hordeconfig) > 1:
            maxhordelen = int(args.hordeconfig[1])
        if len(args.hordeconfig) > 2:
            maxhordectx = int(args.hordeconfig[2])
        if args.debugmode == 0:
            args.debugmode = -1

    if args.debugmode != 1:
        showdebug = False

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
        exitcounter = 999
        print(f"Cannot find model file: {args.model_param}")
        time.sleep(3)
        sys.exit(2)

    if args.lora and args.lora[0]!="":
        if not os.path.exists(args.lora[0]):
            exitcounter = 999
            print(f"Cannot find lora file: {args.lora[0]}")
            time.sleep(3)
            sys.exit(2)
        else:
            args.lora[0] = os.path.abspath(args.lora[0])
            if len(args.lora) > 1:
                if not os.path.exists(args.lora[1]):
                    exitcounter = 999
                    print(f"Cannot find lora base: {args.lora[1]}")
                    time.sleep(3)
                    sys.exit(2)
                else:
                    args.lora[1] = os.path.abspath(args.lora[1])

    if not args.blasthreads or args.blasthreads <= 0:
        args.blasthreads = args.threads

    modelname = os.path.abspath(args.model_param)
    print(args)
    print(f"==========\nLoading model: {modelname} \n[Threads: {args.threads}, BlasThreads: {args.blasthreads}, SmartContext: {args.smartcontext}, ContextShift: {not (args.noshift)}]")
    loadok = load_model(modelname)
    print("Load Model OK: " + str(loadok))

    if not loadok:
        exitcounter = 999
        print("Could not load model: " + modelname)
        time.sleep(3)
        sys.exit(3)
    try:
        basepath = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(basepath, "klite.embd"), mode='rb') as f:
            embedded_kailite = f.read()
            # patch it with extra stuff
            origStr = "Sorry, Kobold Lite requires Javascript to function."
            patchedStr = "Sorry, Kobold Lite requires Javascript to function.<br>You can use <a class=\"color_blueurl\" href=\"/noscript\">KoboldCpp NoScript mode</a> instead."
            embedded_kailite = embedded_kailite.decode("UTF-8","ignore")
            embedded_kailite = embedded_kailite.replace(origStr, patchedStr)
            embedded_kailite = embedded_kailite.encode()
            print("Embedded Kobold Lite loaded.")
    except Exception as e:
        print("Could not find Kobold Lite. Embedded Kobold Lite will not be available.")

    try:
        basepath = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(basepath, "kcpp_docs.embd"), mode='rb') as f:
            embedded_kcpp_docs = f.read()
    except Exception as e:
        print("Could not find Embedded KoboldCpp API docs.")

    if args.port_param!=defaultport:
        args.port = args.port_param

    epurl = ""
    if args.host=="":
        epurl = f"http://localhost:{args.port}"
    else:
        epurl = f"http://{args.host}:{args.port}"
    if not args.remotetunnel:
        print(f"Starting Kobold API on port {args.port} at {epurl}/api/")
        print(f"Starting OpenAI Compatible API on port {args.port} at {epurl}/v1/")

    if args.launch:
        try:
            import webbrowser as wb
            wb.open(epurl)
        except:
            print("--launch was set, but could not launch web browser automatically.")

    if args.hordeconfig and len(args.hordeconfig)>4:
        horde_thread = threading.Thread(target=run_horde_worker,args=(args,args.hordeconfig[3],args.hordeconfig[4]))
        horde_thread.daemon = True
        horde_thread.start()

    #if post-ready script specified, execute it
    if args.onready:
        def onready_subprocess():
            import subprocess
            print("Starting Post-Load subprocess...")
            subprocess.run(args.onready[0], shell=True)
        timer_thread = threading.Timer(1, onready_subprocess) #1 second delay
        timer_thread.start()

    if start_server:
        if args.remotetunnel:
            setuptunnel()
        print(f"======\nPlease connect to custom endpoint at {epurl}")
        asyncio.run(RunServerMultiThreaded(args.host, args.port, embedded_kailite, embedded_kcpp_docs))
    else:
        print(f"Server was not started, main function complete. Idling.")

if __name__ == '__main__':
    print("***\nWelcome to KoboldCpp - Version " + KcppVersion) # just update version manually
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
    parser.add_argument("--lora", help="LLAMA models only, applies a lora file on top of model. Experimental.", metavar=('[lora_filename]', '[lora_base]'), nargs='+')
    parser.add_argument("--config", help="Load settings from a .kcpps file. Other arguments will be ignored", type=str, nargs=1)
    physical_core_limit = 1
    if os.cpu_count()!=None and os.cpu_count()>1:
        physical_core_limit = int(os.cpu_count()/2)
    default_threads = (physical_core_limit if physical_core_limit<=3 else max(3,physical_core_limit-1))
    parser.add_argument("--threads", help="Use a custom number of threads if specified. Otherwise, uses an amount based on CPU cores", type=int, default=default_threads)
    parser.add_argument("--blasthreads", help="Use a different number of threads during BLAS if specified. Otherwise, has the same value as --threads",metavar=('[threads]'), type=int, default=0)
    parser.add_argument("--highpriority", help="Experimental flag. If set, increases the process CPU priority, potentially speeding up generation. Use caution.", action='store_true')
    parser.add_argument("--contextsize", help="Controls the memory allocated for maximum context size, only change if you need more RAM for big contexts. (default 2048)", type=int,choices=[256, 512,1024,2048,3072,4096,6144,8192,12288,16384,24576,32768,65536], default=2048)
    parser.add_argument("--blasbatchsize", help="Sets the batch size used in BLAS processing (default 512). Setting it to -1 disables BLAS mode, but keeps other benefits like GPU offload.", type=int,choices=[-1,32,64,128,256,512,1024,2048], default=512)
    parser.add_argument("--ropeconfig", help="If set, uses customized RoPE scaling from configured frequency scale and frequency base (e.g. --ropeconfig 0.25 10000). Otherwise, uses NTK-Aware scaling set automatically based on context size. For linear rope, simply set the freq-scale and ignore the freq-base",metavar=('[rope-freq-scale]', '[rope-freq-base]'), default=[0.0, 10000.0], type=float, nargs='+')
    parser.add_argument("--smartcontext", help="Reserving a portion of context to try processing less frequently.", action='store_true')
    parser.add_argument("--noshift", help="If set, do not attempt to Trim and Shift the GGUF context.", action='store_true')
    parser.add_argument("--bantokens", help="You can manually specify a list of token SUBSTRINGS that the AI cannot use. This bans ALL instances of that substring.", metavar=('[token_substrings]'), nargs='+')
    parser.add_argument("--forceversion", help="If the model file format detection fails (e.g. rogue modified model) you can set this to override the detected format (enter desired version, e.g. 401 for GPTNeoX-Type2).",metavar=('[version]'), type=int, default=0)
    parser.add_argument("--nommap", help="If set, do not use mmap to load newer models", action='store_true')
    parser.add_argument("--usemlock", help="For Apple Systems. Force system to keep model in RAM rather than swapping or compressing", action='store_true')
    parser.add_argument("--noavx2", help="Do not use AVX2 instructions, a slower compatibility mode for older devices. Does not work with --clblast.", action='store_true')
    parser.add_argument("--debugmode", help="Shows additional debug info in the terminal.", nargs='?', const=1, type=int, default=0)
    parser.add_argument("--skiplauncher", help="Doesn't display or use the GUI launcher.", action='store_true')
    parser.add_argument("--hordeconfig", help="Sets the display model name to something else, for easy use on AI Horde. Optional additional parameters set the horde max genlength, max ctxlen, API key and worker name.",metavar=('[hordemodelname]', '[hordegenlength] [hordemaxctx] [hordeapikey] [hordeworkername]'), nargs='+')
    compatgroup = parser.add_mutually_exclusive_group()
    compatgroup.add_argument("--noblas", help="Do not use OpenBLAS for accelerated prompt ingestion", action='store_true')
    compatgroup.add_argument("--useclblast", help="Use CLBlast for GPU Acceleration. Must specify exactly 2 arguments, platform ID and device ID (e.g. --useclblast 1 0).", type=int, choices=range(0,9), nargs=2)
    compatgroup.add_argument("--usecublas", help="Use CuBLAS for GPU Acceleration. Requires CUDA. Select lowvram to not allocate VRAM scratch buffer. Enter a number afterwards to select and use 1 GPU. Leaving no number will use all GPUs. For hipBLAS binaries, please check YellowRoseCx rocm fork.", nargs='*',metavar=('[lowvram|normal] [main GPU ID] [mmq]'), choices=['normal', 'lowvram', '0', '1', '2', '3', 'mmq'])
    parser.add_argument("--gpulayers", help="Set number of layers to offload to GPU when using GPU. Requires GPU.",metavar=('[GPU layers]'), type=int, default=0)
    parser.add_argument("--tensor_split", help="For CUDA with ALL GPU set only, ratio to split tensors across multiple GPUs, space-separated list of proportions, e.g. 7 3", metavar=('[Ratios]'), type=float, nargs='+')
    parser.add_argument("--onready", help="An optional shell command to execute after the model has been loaded.", metavar=('[shell command]'), type=str, default="",nargs=1)
    parser.add_argument("--multiuser", help="Runs in multiuser mode, which queues incoming requests instead of blocking them.", metavar=('limit'), nargs='?', const=1, type=int, default=0)
    parser.add_argument("--remotetunnel", help="Uses Cloudflare to create a remote tunnel, allowing you to access koboldcpp remotely over the internet even behind a firewall.", action='store_true')
    parser.add_argument("--foreground", help="Windows only. Sends the terminal to the foreground every time a new prompt is generated. This helps avoid some idle slowdown issues.", action='store_true')
    parser.add_argument("--preloadstory", help="Configures a prepared story json save file to be hosted on the server, which frontends (such as Kobold Lite) can access over the API.", default="")
    parser.add_argument("--quiet", help="Enable quiet mode, which hides generation inputs and outputs in the terminal. Quiet mode is automatically enabled when running --hordeconfig.", action='store_true')

    # #deprecated hidden args. they do nothing. do not use
    # parser.add_argument("--psutil_set_threads", action='store_true', help=argparse.SUPPRESS)
    # parser.add_argument("--stream", action='store_true', help=argparse.SUPPRESS)
    # parser.add_argument("--unbantokens", action='store_true', help=argparse.SUPPRESS)
    # parser.add_argument("--usemirostat", action='store_true', help=argparse.SUPPRESS)

    main(parser.parse_args(),start_server=True)
