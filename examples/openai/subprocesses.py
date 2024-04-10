
import atexit
import os
import signal
import subprocess
import sys


def _cleanup_process(p):
    pid = p.pid

    if sys.platform == 'win32':
        os.system(f'taskkill /PID {pid} /T /F')
    else:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)

        p.wait()
        if p.poll() is None:
            os.killpg(pgid, signal.SIGKILL)

def spawn_subprocess(cmd, **kwargs):
    server_process = subprocess.Popen(
        cmd,
        stdout=sys.stderr,
        start_new_session=True,
        **kwargs
    )
    atexit.register(_cleanup_process, server_process)
    return server_process
