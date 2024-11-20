import pytest
from openai import OpenAI
from utils import *

server = ServerPreset.bert_bge_small()


@pytest.fixture(scope="module", autouse=True)
def create_server():
    global server
    server = ServerPreset.bert_bge_small()


def test_embedding_single():
    global server
    server.start()
    res = server.make_request("POST", "/embeddings", data={
        "input": "I believe the meaning of life is",
    })
    assert res.status_code == 200
    assert len(res.body['data']) == 1
    assert 'embedding' in res.body['data'][0]
    assert len(res.body['data'][0]['embedding']) > 1

    # make sure embedding vector is normalized
    assert abs(sum([x ** 2 for x in res.body['data'][0]['embedding']]) - 1) < 1e-5


def test_embedding_multiple():
    global server
    server.start()
    res = server.make_request("POST", "/embeddings", data={
        "input": [
            "I believe the meaning of life is",
            "Write a joke about AI from a very long prompt which will not be truncated",
            "This is a test",
            "This is another test",
        ],
    })
    assert res.status_code == 200
    assert len(res.body['data']) == 4
    for d in res.body['data']:
        assert 'embedding' in d
        assert len(d['embedding']) > 1


def test_embedding_openai_library_single():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}")
    res = client.embeddings.create(model="text-embedding-3-small", input="I believe the meaning of life is")
    assert len(res.data) == 1
    assert len(res.data[0].embedding) > 1


def test_embedding_openai_library_multiple():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}")
    res = client.embeddings.create(model="text-embedding-3-small", input=[
        "I believe the meaning of life is",
        "Write a joke about AI from a very long prompt which will not be truncated",
        "This is a test",
        "This is another test",
    ])
    assert len(res.data) == 4
    for d in res.data:
        assert len(d.embedding) > 1


def test_embedding_error_prompt_too_long():
    global server
    server.start()
    res = server.make_request("POST", "/embeddings", data={
        "input": "This is a test " * 512,
    })
    assert res.status_code != 200
    assert "too large" in res.body["error"]["message"]


def test_same_prompt_give_same_result():
    server.start()
    res = server.make_request("POST", "/embeddings", data={
        "input": [
            "I believe the meaning of life is",
            "I believe the meaning of life is",
            "I believe the meaning of life is",
            "I believe the meaning of life is",
            "I believe the meaning of life is",
        ],
    })
    assert res.status_code == 200
    assert len(res.body['data']) == 5
    for i in range(1, len(res.body['data'])):
        v0 = res.body['data'][0]['embedding']
        vi = res.body['data'][i]['embedding']
        for x, y in zip(v0, vi):
            assert abs(x - y) < 1e-5
