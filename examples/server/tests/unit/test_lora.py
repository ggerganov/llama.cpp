import pytest
import os
from utils import *

server = ServerPreset.stories15m_moe()

LORA_FILE_URL = "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf"

@pytest.fixture(scope="module", autouse=True)
def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    # download lora file if needed
    file_name = LORA_FILE_URL.split('/').pop()
    lora_file = f'../../../{file_name}'
    if not os.path.exists(lora_file):
        print(f"Downloading {LORA_FILE_URL} to {lora_file}")
        with open(lora_file, 'wb') as f:
            f.write(requests.get(LORA_FILE_URL).content)
        print(f"Done downloading lora file")
    server.lora_files = [lora_file]


@pytest.mark.parametrize("scale,re_content", [
    # without applying lora, the model should behave like a bedtime story generator
    (0.0, "(little|girl|three|years|old)+"),
    # with lora, the model should behave like a Shakespearean text generator
    (1.0, "(eye|love|glass|sun)+"),
])
def test_lora(scale: float, re_content: str):
    global server
    server.start()
    res_lora_control = server.make_request("POST", "/lora-adapters", data=[
        {"id": 0, "scale": scale}
    ])
    assert res_lora_control.status_code == 200
    res = server.make_request("POST", "/completion", data={
        "prompt": "Look in thy glass",
    })
    assert res.status_code == 200
    assert match_regex(re_content, res.body["content"])

