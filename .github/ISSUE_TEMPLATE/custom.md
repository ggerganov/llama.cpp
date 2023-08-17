---
name: Issue and enhancement template
about: Used to report issues and request enhancements for llama.cpp
title: "[User] Insert summary of your issue or enhancement.."
labels: ''
assignees: ''

---

# Prerequisites

Please answer the following questions for yourself before submitting an issue.

- [ ] I am running the latest code. Development is very rapid so there are no tagged versions as of now.
- [ ] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [ ] I [searched using keywords relevant to my issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/filtering-and-searching-issues-and-pull-requests) to make sure that I am creating a new issue that is not already open (or closed).
- [ ] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new bug or useful enhancement to share.

# Expected Behavior

Please provide a detailed written description of what you were trying to do, and what you expected `llama.cpp` to do.

# Current Behavior

Please provide a detailed written description of what `llama.cpp` did, instead.

# Environment and Context

Please provide detailed information about your computer setup. This is important in case the issue is not reproducible except for under certain specific conditions.

* Physical (or virtual) hardware you are using, e.g. for Linux:

`$ lscpu`

* Operating System, e.g. for Linux:

`$ uname -a`

* SDK version, e.g. for Linux:

```
$ python3 --version
$ make --version
$ g++ --version
```

# Failure Information (for bugs)

Please help provide information about the failure if this is a bug. If it is not a bug, please remove the rest of this template.

# Steps to Reproduce

Please provide detailed steps for reproducing the issue. We are not sitting in front of your screen, so the more detail the better.

1. step 1
2. step 2
3. step 3
4. etc.

# Failure Logs

Please include any relevant log snippets or files. If it works under one configuration but not under another, please provide logs for both configurations and their corresponding outputs so it is easy to see where behavior changes.

Also, please try to **avoid using screenshots** if at all possible. Instead, copy/paste the console output and use [Github's markdown](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) to cleanly format your logs for easy readability.

Example environment info:
```
llama.cpp$ git log | head -1
commit 2af23d30434a677c6416812eea52ccc0af65119c

llama.cpp$ lscpu | egrep "AMD|Flags"
Vendor ID:                       AuthenticAMD
Model name:                      AMD Ryzen Threadripper 1950X 16-Core Processor
Flags:                           fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid amd_dcm aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb hw_pstate ssbd ibpb vmmcall fsgsbase bmi1 avx2 smep bmi2 rdseed adx smap clflushopt sha_ni xsaveopt xsavec xgetbv1 xsaves clzero irperf xsaveerptr arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif overflow_recov succor smca sme sev
Virtualization:                  AMD-V

llama.cpp$ python3 --version
Python 3.10.9

llama.cpp$ pip list | egrep "torch|numpy|sentencepiece"
numpy                         1.24.2
numpydoc                      1.5.0
sentencepiece                 0.1.97
torch                         1.13.1
torchvision                   0.14.1

llama.cpp$ make --version | head -1
GNU Make 4.3

$ md5sum ./models/65B/ggml-model-q4_0.bin
dbdd682cce80e2d6e93cefc7449df487  ./models/65B/ggml-model-q4_0.bin
```

Example run with the Linux command [perf](https://www.brendangregg.com/perf.html)
```
llama.cpp$ perf stat ./main -m ./models/65B/ggml-model-q4_0.bin -t 16 -n 1024 -p "Please close your issue when it has been answered."
main: seed = 1679149377
llama_model_load: loading model from './models/65B/ggml-model-q4_0.bin' - please wait ...
llama_model_load: n_vocab = 32000
llama_model_load: n_ctx   = 512
llama_model_load: n_embd  = 8192
llama_model_load: n_mult  = 256
llama_model_load: n_head  = 64
llama_model_load: n_layer = 80
llama_model_load: n_rot   = 128
llama_model_load: f16     = 2
llama_model_load: n_ff    = 22016
llama_model_load: n_parts = 8
llama_model_load: ggml ctx size = 41477.73 MB
llama_model_load: memory_size =  2560.00 MB, n_mem = 40960
llama_model_load: loading model part 1/8 from './models/65B/ggml-model-q4_0.bin'
llama_model_load: .......................................................................................... done
llama_model_load: model size =  4869.09 MB / num tensors = 723
llama_model_load: loading model part 2/8 from './models/65B/ggml-model-q4_0.bin.1'
llama_model_load: .......................................................................................... done
llama_model_load: model size =  4869.09 MB / num tensors = 723
llama_model_load: loading model part 3/8 from './models/65B/ggml-model-q4_0.bin.2'
llama_model_load: .......................................................................................... done
llama_model_load: model size =  4869.09 MB / num tensors = 723
llama_model_load: loading model part 4/8 from './models/65B/ggml-model-q4_0.bin.3'
llama_model_load: .......................................................................................... done
llama_model_load: model size =  4869.09 MB / num tensors = 723
llama_model_load: loading model part 5/8 from './models/65B/ggml-model-q4_0.bin.4'
llama_model_load: .......................................................................................... done
llama_model_load: model size =  4869.09 MB / num tensors = 723
llama_model_load: loading model part 6/8 from './models/65B/ggml-model-q4_0.bin.5'
llama_model_load: .......................................................................................... done
llama_model_load: model size =  4869.09 MB / num tensors = 723
llama_model_load: loading model part 7/8 from './models/65B/ggml-model-q4_0.bin.6'
llama_model_load: .......................................................................................... done
llama_model_load: model size =  4869.09 MB / num tensors = 723
llama_model_load: loading model part 8/8 from './models/65B/ggml-model-q4_0.bin.7'
llama_model_load: .......................................................................................... done
llama_model_load: model size =  4869.09 MB / num tensors = 723

system_info: n_threads = 16 / 32 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | VSX = 0 |

main: prompt: 'Please close your issue when it has been answered.'
main: number of tokens in prompt = 11
     1 -> ''
 12148 -> 'Please'
  3802 -> ' close'
   596 -> ' your'
  2228 -> ' issue'
   746 -> ' when'
   372 -> ' it'
   756 -> ' has'
  1063 -> ' been'
  7699 -> ' answered'
 29889 -> '.'

sampling parameters: temp = 0.800000, top_k = 40, top_p = 0.950000, repeat_last_n = 64, repeat_penalty = 1.300000


Please close your issue when it has been answered.
@duncan-donut: I'm trying to figure out what kind of "support" you need for this script and why, exactly? Is there a question about how the code works that hasn't already been addressed in one or more comments below this ticket, or are we talking something else entirely like some sorta bugfixing job because your server setup is different from mine??
I can understand if your site needs to be running smoothly and you need help with a fix of sorts but there should really be nothing wrong here that the code itself could not handle. And given that I'm getting reports about how it works perfectly well on some other servers, what exactly are we talking? A detailed report will do wonders in helping us get this resolved for ya quickly so please take your time and describe the issue(s) you see as clearly & concisely as possible!!
@duncan-donut: I'm not sure if you have access to cPanel but you could try these instructions. It is worth a shot! Let me know how it goes (or what error message, exactly!) when/if ya give that code a go? [end of text]


main: mem per token = 71159620 bytes
main:     load time = 19309.95 ms
main:   sample time =   168.62 ms
main:  predict time = 223895.61 ms / 888.47 ms per token
main:    total time = 246406.42 ms

 Performance counter stats for './main -m ./models/65B/ggml-model-q4_0.bin -t 16 -n 1024 -p Please close your issue when it has been answered.':

        3636882.89 msec task-clock                #   14.677 CPUs utilized
             13509      context-switches          #    3.714 /sec
              2436      cpu-migrations            #    0.670 /sec
          10476679      page-faults               #    2.881 K/sec
    13133115082869      cycles                    #    3.611 GHz                      (16.77%)
       29314462753      stalled-cycles-frontend   #    0.22% frontend cycles idle     (16.76%)
    10294402631459      stalled-cycles-backend    #   78.39% backend cycles idle      (16.74%)
    23479217109614      instructions              #    1.79  insn per cycle
                                                  #    0.44  stalled cycles per insn  (16.76%)
     2353072268027      branches                  #  647.002 M/sec                    (16.77%)
        1998682780      branch-misses             #    0.08% of all branches          (16.76%)

     247.802177522 seconds time elapsed

    3618.573072000 seconds user
      18.491698000 seconds sys
```
