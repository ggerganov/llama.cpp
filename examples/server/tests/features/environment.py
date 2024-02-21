
def after_scenario(context, scenario):
    print("stopping server...")
    context.server_process.kill()
