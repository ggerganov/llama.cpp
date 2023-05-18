#!/usr/bin/env python3

import wasm2kast
import json
import pyk
import sys
import subprocess
import tempfile

WASM_definition_llvm_no_coverage_dir = '.build/defn/llvm'

def config_to_kast_term(config):
    return { 'format' : 'KAST', 'version': 2, 'term': config.to_dict() }

def run_module(parsed_module):
    input_json = config_to_kast_term(parsed_module)
    krun_args = [ '--term', '--debug']

    with tempfile.NamedTemporaryFile(mode = 'w') as tempf:
        tempf.write(json.dumps(input_json))
        tempf.flush()

        command = ['krun-legacy', '--directory',
                WASM_definition_llvm_no_coverage_dir, '--term', tempf.name,
                '--parser', 'cat', '--output', 'json']
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(input=None)
        rc = process.returncode

        if rc != 0:
            raise Exception(f'Received error while running: {str(stderr)}')

def pykPrettyPrint(module):
    WASM_definition_llvm_no_coverage_dir = '.build/defn/llvm'
    WASM_definition_main_file = 'test'
    WASM_definition_llvm_no_coverage = pyk.readKastTerm(WASM_definition_llvm_no_coverage_dir + '/' + WASM_definition_main_file + '-kompiled/compiled.json')
    WASM_symbols_llvm_no_coverage = pyk.buildSymbolTable(WASM_definition_llvm_no_coverage)
    print(pyk.prettyPrintKast(module, WASM_symbols_llvm_no_coverage))

sys.setrecursionlimit(1500000000)

if __name__ == "__main__":
    module = wasm2kast.main()
    run_module(module)
