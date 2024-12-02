import os


def compile_source(source):
    open("tmp_kp_shader.comp", "w").write(source)
    os.system("glslangValidator -V tmp_kp_shader.comp -o tmp_kp_shader.comp.spv")
    return open("tmp_kp_shader.comp.spv", "rb").read()
