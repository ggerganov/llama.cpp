import pytest
from utils import *

server = ServerPreset.tinyllamas()


@pytest.fixture(scope="module", autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllamas()


def test_tokenize_detokenize():
    global server
    server.start()
    # tokenize
    content = "What is the capital of France ?"
    resTok = server.make_request("POST", "/tokenize", data={
        "content": content
    })
    assert resTok.status_code == 200
    assert len(resTok.body["tokens"]) > 5
    # detokenize
    resDetok = server.make_request("POST", "/detokenize", data={
        "tokens": resTok.body["tokens"],
    })
    assert resDetok.status_code == 200
    assert resDetok.body["content"].strip() == content


def test_tokenize_with_bos():
    global server
    server.start()
    # tokenize
    content = "What is the capital of France ?"
    bosId = 1
    resTok = server.make_request("POST", "/tokenize", data={
        "content": content,
        "add_special": True,
    })
    assert resTok.status_code == 200
    assert resTok.body["tokens"][0] == bosId


def test_tokenize_with_pieces():
    global server
    server.start()
    # tokenize
    content = "This is a test string with unicode 媽 and emoji 🤗"
    resTok = server.make_request("POST", "/tokenize", data={
        "content": content,
        "with_pieces": True,
    })
    assert resTok.status_code == 200
    for token in resTok.body["tokens"]:
        assert "id" in token
        assert token["id"] > 0
        assert "piece" in token
        assert len(token["piece"]) > 0
