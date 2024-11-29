import pytest
from utils import *

server = ServerPreset.tinyllama_infill()

@pytest.fixture(scope="module", autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama_infill()

def test_infill_without_input_extra():
    global server
    server.start()
    res = server.make_request("POST", "/infill", data={
        "prompt": "Complete this",
        "input_prefix": "#include <cstdio>\n#include \"llama.h\"\n\nint main() {\n    int n_threads = llama_",
        "input_suffix": "}\n",
    })
    assert res.status_code == 200
    assert match_regex("(One|day|she|saw|big|scary|bird)+", res.body["content"])

def test_infill_with_input_extra():
    global server
    server.start()
    res = server.make_request("POST", "/infill", data={
        "prompt": "Complete this",
        "input_extra": [{
            "filename": "llama.h",
            "text": "LLAMA_API int32_t llama_n_threads();\n"
        }],
        "input_prefix": "#include <cstdio>\n#include \"llama.h\"\n\nint main() {\n    int n_threads = llama_",
        "input_suffix": "}\n",
    })
    assert res.status_code == 200
    assert match_regex("(cuts|Jimmy|mom|came|into|the|room)+", res.body["content"])
