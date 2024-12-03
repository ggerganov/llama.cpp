import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(scope="module", autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def test_server_start_simple():
    global server
    server.start()
    res = server.make_request("GET", "/health")
    assert res.status_code == 200


def test_server_props():
    global server
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body["total_slots"] == server.n_slots


def test_server_models():
    global server
    server.start()
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    assert len(res.body["data"]) == 1
    assert res.body["data"][0]["id"] == server.model_alias

def test_load_split_model():
    global server
    server.model_hf_repo = "ggml-org/models"
    server.model_hf_file = "tinyllamas/split/stories15M-q8_0-00001-of-00003.gguf"
    server.model_alias = "tinyllama-split"
    server.start()
    res = server.make_request("POST", "/completion", data={
        "max_tokens": 16,
        "prompt": "Hello",
        "temperature": 0.0,
    })
    assert res.status_code == 200
    # Verify response structure
    assert "id" in res.body
    assert "object" in res.body
    assert "created" in res.body
    assert "model" in res.body
    assert "choices" in res.body
    assert isinstance(res.body["choices"], list)
    assert len(res.body["choices"]) > 0
    assert "text" in res.body["choices"][0]
    # Verify the actual content
    assert match_regex("(little|girl)+", res.body["choices"][0]["text"])
