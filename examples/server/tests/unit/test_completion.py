import pytest
from openai import OpenAI
from utils import *

server = ServerPreset.tinyllamas()


@pytest.fixture(scope="module", autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllamas()


@pytest.mark.parametrize("prompt,n_predict,re_content,n_prompt,n_predicted,truncated", [
    ("I believe the meaning of life is", 8, "(going|bed)+", 18, 8, False),
    ("Write a joke about AI from a very long prompt which will not be truncated", 256, "(princesses|everyone|kids|Anna|forest)+", 46, 64, False),
])
def test_completion(prompt: str, n_predict: int, re_content: str, n_prompt: int, n_predicted: int, truncated: bool):
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": n_predict,
        "prompt": prompt,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == n_prompt
    assert res.body["timings"]["predicted_n"] == n_predicted
    assert res.body["truncated"] == truncated
    assert match_regex(re_content, res.body["content"])


@pytest.mark.parametrize("prompt,n_predict,re_content,n_prompt,n_predicted,truncated", [
    ("I believe the meaning of life is", 8, "(going|bed)+", 18, 8, False),
    ("Write a joke about AI from a very long prompt which will not be truncated", 256, "(princesses|everyone|kids|Anna|forest)+", 46, 64, False),
])
def test_completion_stream(prompt: str, n_predict: int, re_content: str, n_prompt: int, n_predicted: int, truncated: bool):
    global server
    server.start()
    res = server.make_stream_request("POST", "/completion", data={
        "n_predict": n_predict,
        "prompt": prompt,
        "stream": True,
    })
    content = ""
    for data in res:
        if data["stop"]:
            assert data["timings"]["prompt_n"] == n_prompt
            assert data["timings"]["predicted_n"] == n_predicted
            assert data["truncated"] == truncated
            assert match_regex(re_content, content)
        else:
            content += data["content"]


# FIXME: This test is not working because /completions endpoint is not OAI-compatible
@pytest.mark.skip(reason="Only /chat/completions is OAI-compatible for now")
def test_completion_with_openai_library():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}")
    res = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt="I believe the meaning of life is",
        n=8,
        seed=42,
        temperature=0.8,
    )
    print(res)
    assert res.choices[0].finish_reason == "length"
    assert match_regex("(going|bed)+", res.choices[0].text)
