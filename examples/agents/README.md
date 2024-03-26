
Edit `examples/agents/hermes_function_calling/utils.py`:

```py
log_folder = os.environ.get('LOG_FOLDER', os.path.join(script_dir, "inference_logs"))
```

Then run:

```bash
REQUIREMENTS_FILE=<( cat examples/agents/hermes_function_calling/requirements.txt | grep -vE "bitsandbytes|flash-attn" ) \
  examples/agents/run_sandboxed_tools.sh \
    examples/agents/hermes_function_calling/functions.py \
    -e LOG_FOLDER=/data/inference_logs
```