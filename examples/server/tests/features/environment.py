import os
import socket
import subprocess
import time
from contextlib import closing
from signal import SIGKILL


def before_scenario(context, scenario):
    context.debug = 'DEBUG' in os.environ and os.environ['DEBUG'] == 'ON'
    if context.debug:
        print("DEBUG=ON\n")
    print(f"\x1b[33;42mStarting new scenario: {scenario.name}!\x1b[0m\n")
    port = 8080
    if 'PORT' in os.environ:
        port = int(os.environ['PORT'])
    if is_server_listening("localhost", port):
        assert False, "Server already started"


def after_scenario(context, scenario):
    if context.server_process is None:
        return
    if scenario.status == "failed":
        if 'GITHUB_ACTIONS' in os.environ:
            print(f"\x1b[33;101mSCENARIO FAILED: {scenario.name} server logs:\x1b[0m\n\n")
            if os.path.isfile('llama.log'):
                with closing(open('llama.log', 'r')) as f:
                    for line in f:
                        print(line)
        if not is_server_listening(context.server_fqdn, context.server_port):
            print("\x1b[33;101mERROR: Server stopped listening\x1b[0m")

    if not pid_exists(context.server_process.pid):
        assert False, f"Server not running pid={context.server_process.pid} ..."

    print(f"stopping server pid={context.server_process.pid} ...")
    context.server_process.kill()
    # Wait few for socket to free up
    time.sleep(0.05)

    attempts = 0
    while is_server_listening(context.server_fqdn, context.server_port):
        print(f"stopping server pid={context.server_process.pid} ...")
        os.kill(context.server_process.pid, SIGKILL)
        time.sleep(0.1)
        attempts += 1
        if attempts > 5:
            print(f"Server dangling exits, killing all {context.server_path} ...")
            process = subprocess.run(['killall', '-9', context.server_path],
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)
            print(process)


def is_server_listening(server_fqdn, server_port):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        result = sock.connect_ex((server_fqdn, server_port))
        return result == 0


def pid_exists(pid):
    """Check whether pid exists in the current process table."""
    import errno
    if pid < 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as e:
        return e.errno == errno.EPERM
    else:
        return True
