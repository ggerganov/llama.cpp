import os
import signal
import socket
import sys
import time
import traceback
from contextlib import closing

import psutil


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
    try:
        if 'server_process' not in context or context.server_process is None:
            return
        if scenario.status == "failed":
            if 'GITHUB_ACTIONS' in os.environ:
                print(f"\x1b[33;101mSCENARIO FAILED: {scenario.name} server logs:\x1b[0m\n\n")
                if os.path.isfile('llama.log'):
                    with closing(open('llama.log', 'r')) as f:
                        for line in f:
                            print(line)
            if not is_server_listening(context.server_fqdn, context.server_port):
                print("\x1b[33;101mERROR: Server stopped listening\x1b[0m\n")

        if not pid_exists(context.server_process.pid):
            assert False, f"Server not running pid={context.server_process.pid} ..."

        server_graceful_shutdown(context)

        # Wait few for socket to free up
        time.sleep(0.05)

        attempts = 0
        while pid_exists(context.server_process.pid) or is_server_listening(context.server_fqdn, context.server_port):
            server_kill(context)
            time.sleep(0.1)
            attempts += 1
            if attempts > 5:
                server_kill_hard(context)
    except:
        exc = sys.exception()
        print("error in after scenario: \n")
        print(exc)
        print("*** print_tb: \n")
        traceback.print_tb(exc.__traceback__, file=sys.stdout)


def server_graceful_shutdown(context):
    print(f"shutting down server pid={context.server_process.pid} ...\n")
    if os.name == 'nt':
        os.kill(context.server_process.pid, signal.CTRL_C_EVENT)
    else:
        os.kill(context.server_process.pid, signal.SIGINT)


def server_kill(context):
    print(f"killing server pid={context.server_process.pid} ...\n")
    context.server_process.kill()


def server_kill_hard(context):
    pid = context.server_process.pid
    path = context.server_path

    print(f"Server dangling exits, hard killing force {pid}={path}...\n")
    try:
        psutil.Process(pid).kill()
    except psutil.NoSuchProcess:
        return False
    return True


def is_server_listening(server_fqdn, server_port):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        result = sock.connect_ex((server_fqdn, server_port))
        _is_server_listening = result == 0
        if _is_server_listening:
            print(f"server is listening on {server_fqdn}:{server_port}...\n")
        return _is_server_listening


def pid_exists(pid):
    try:
        psutil.Process(pid)
    except psutil.NoSuchProcess:
        return False
    return True

