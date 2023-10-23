#!/usr/bin/env python3
"""
    Script to handle conversion of compute shaders to spirv and to headers
"""
import os
import sys
import logging
import click
import subprocess

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

is_windows = sys.platform.startswith('win')

CWD=os.path.dirname(os.path.abspath(__file__))
XXD_LINUX_CMD="xxd"
XXD_WINDOWS_CMD=os.path.abspath(os.path.join(CWD, "..\\external\\bin\\", "xxd.exe"))

SHADER_GENERATED_NOTICE = """/*
    THIS FILE HAS BEEN AUTOMATICALLY GENERATED - DO NOT EDIT

    ---

    Copyright 2020 The Institute for Ethical AI & Machine Learning

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/
"""

@click.command()
@click.option(
    "--shader-path",
    "-p",
    envvar="KOMPUTE_SHADER_PATH",
    required=True,
    help="The path for the directory to build and convert shaders",
)
@click.option(
    "--shader-binary",
    "-s",
    envvar="KOMPUTE_SHADER_BINARY",
    required=True,
    help="The path for the directory to build and convert shaders",
)
@click.option(
    "--header-path",
    "-c",
    envvar="KOMPUTE_HEADER_PATH",
    default="",
    required=False,
    help="The (optional) output file for the cpp header files",
)
@click.option(
    "--verbose",
    "-v",
    envvar="KOMPUTE_HEADER_PATH",
    default=False,
    is_flag=True,
    help="Enable versbosity if flag is provided",
)
def run_cli(
    shader_path: str = None,
    shader_binary: str = None,
    header_path: bool = None,
    verbose: bool = None,
):
    """
    CLI function for shader generation
    """

    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    logger.debug(f"Starting script with variables: {locals()}")

    if is_windows:
        logger.debug(f"Running on windows, converting input paths")
        shader_path = shader_path.replace("/", "\\")
        header_path = header_path.replace("/", "\\")

    shader_files = []
    for root, directory, files in os.walk(shader_path):
        for file in files:
            if file.endswith(".comp"):
                shader_files.append(os.path.join(root, file))

    run_cmd = lambda *args: subprocess.check_output([*args]).decode()

    logger.debug(f"Output spirv path: {shader_path}")
    logger.debug(f"Converting files to spirv: {shader_files}")

    spirv_files = []
    for file in shader_files:
        logger.debug(f"Converting to spirv: {file}")
        spirv_file = f"{file}.spv"
        run_cmd(shader_binary, "-V", file, "-o", spirv_file)
        spirv_files.append(spirv_file)

    # Create cpp files if header_path provided
    if header_path:
        logger.debug(f"Header path provided. Converting bin files to hpp.")
        logger.debug(f"Output header path: {shader_path}")

        # Check if xxd command options are available
        if is_windows:
            xxd_cmd = XXD_WINDOWS_CMD
        else:
            xxd_cmd = XXD_LINUX_CMD

        for file in spirv_files:
            print(xxd_cmd)
            header_data = str(run_cmd(xxd_cmd, "-i", file))
            # Ensuring the variable is a static const unsigned
            header_data = header_data.replace("unsigned", "static const unsigned")
            if is_windows:
                raw_file_name = file.split("\\")[-1]
            else:
                raw_file_name = file.split("/")[-1]
            file_name = f"shader{raw_file_name}"
            header_file = file_name.replace(".comp.spv", ".hpp")
            header_file_define = "SHADEROP_" + header_file.replace(".", "_").upper()
            logger.debug(f"Converting to hpp: {file_name}")
            with open(os.path.join(header_path, header_file), "w+", newline='\n') as fstream:
                fstream.write(f"{SHADER_GENERATED_NOTICE}\n")
                fstream.write(f"#ifndef {header_file_define}\n")
                fstream.write(f"#define {header_file_define}\n\n")
                fstream.write("namespace kp {\n")
                fstream.write("namespace shader_data {\n")
                fstream.write(f"{header_data}")
                fstream.write("}\n")
                fstream.write("}\n")
                fstream.write(f"#endif // define {header_file_define}\n")


if __name__ == "__main__":
    run_cli()
