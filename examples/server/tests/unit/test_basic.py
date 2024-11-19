import pytest
from utils import *

server = ServerProcess()


@pytest.fixture(scope="module", autouse=True)
def create_server():
    global server
    server = ServerProcess()
    server.model_hf_repo = "ggml-org/models"
    server.model_hf_file = "tinyllamas/stories260K.gguf"
    server.n_ctx = 256
    server.n_batch = 32
    server.n_slots = 2
    server.n_predict = 64


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
