import multiprocessing
import os
import socket
import subprocess
import time
from contextlib import closing
from signal import SIGKILL


def before_scenario(context, scenario):
    port = 8080
    if 'PORT' in os.environ:
        port = int(os.environ['PORT'])
    if is_server_listening("localhost", port):
        assert False, "Server already started"


def after_scenario(context, scenario):
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
            print(f"Server dandling exits, killing all {context.server_path} ...")
            process = subprocess.run(['killall', '-9', context.server_path],
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)
            print(process)


def is_server_listening(server_fqdn, server_port):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        result = sock.connect_ex((server_fqdn, server_port))
        return result == 0
