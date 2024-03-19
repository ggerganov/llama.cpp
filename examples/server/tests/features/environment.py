import os
import signal
import socket
import sys
import time
import traceback
from contextlib import closing
from subprocess import TimeoutExpired


def before_scenario(context, scenario):
    context.debug = 'DEBUG' in os.environ and os.environ['DEBUG'] == 'ON'
    if context.debug:
        print("DEBUG=ON")
    print(f"\x1b[33;42mStarting new scenario: {scenario.name}!\x1b[0m")
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
                print(f"\x1b[33;101mSCENARIO FAILED: {scenario.name} server logs:\x1b[0m\n")
                if os.path.isfile('llama.log'):
                    with closing(open('llama.log', 'r')) as f:
                        for line in f:
                            print(line)
            if not is_server_listening(context.server_fqdn, context.server_port):
                print("\x1b[33;101mERROR: Server stopped listening\x1b[0m")

        if context.server_process.poll() is not None:
            assert False, f"Server not running pid={context.server_process.pid} ..."

        server_graceful_shutdown(context)  # SIGINT

        try:
            context.server_process.wait(0.5)
        except TimeoutExpired:
            print(f"server still alive after 500ms, force-killing pid={context.server_process.pid} ...")
            context.server_process.kill()  # SIGKILL
            context.server_process.wait()

        while is_server_listening(context.server_fqdn, context.server_port):
            time.sleep(0.1)
    except Exception:
        print("ignoring error in after_scenario:")
        traceback.print_exc(file=sys.stdout)


def server_graceful_shutdown(context):
    print(f"shutting down server pid={context.server_process.pid} ...")
    if os.name == 'nt':
        interrupt = signal.CTRL_C_EVENT
    else:
        interrupt = signal.SIGINT
    context.server_process.send_signal(interrupt)


def is_server_listening(server_fqdn, server_port):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        result = sock.connect_ex((server_fqdn, server_port))
        _is_server_listening = result == 0
        if _is_server_listening:
            print(f"server is listening on {server_fqdn}:{server_port}...")
        return _is_server_listening
