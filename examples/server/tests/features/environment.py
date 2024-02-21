import time


def after_scenario(context, scenario):
    print("stopping server...")
    context.server_process.kill()
    # Wait few for socket to be free up
    time.sleep(0.05)
