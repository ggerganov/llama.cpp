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
