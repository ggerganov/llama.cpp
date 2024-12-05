import pytest
import time
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(scope="module", autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()

@pytest.mark.parametrize("prompt,n_predict,re_content,n_prompt,n_predicted,truncated", [
    ("I believe the meaning of life is", 8, "(going|bed)+", 18, 8, False),
    ("Write a joke about AI from a very long prompt which will not be truncated", 256, "(princesses|everyone|kids|Anna|forest)+", 46, 64, False),
])
def test_completion(prompt: str, n_predict: int, re_content: str, n_prompt: int, n_predicted: int, truncated: bool):
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "max_tokens": n_predict,
        "prompt": prompt,
        "oai_compat": False,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == n_prompt
    assert res.body["timings"]["predicted_n"] == n_predicted
    assert match_regex(re_content, res.body["content"])


@pytest.mark.parametrize("prompt,n_predict,re_content,n_prompt,n_predicted,truncated", [
    ("I believe the meaning of life is", 8, "(going|bed)+", 18, 8, False),
    ("Write a joke about AI from a very long prompt which will not be truncated", 256, "(princesses|everyone|kids|Anna|forest)+", 46, 64, False),
])
def test_completion_stream(prompt: str, n_predict: int, re_content: str, n_prompt: int, n_predicted: int, truncated: bool):
    global server
    server.start()
    res = server.make_stream_request("POST", "/completion", data={
        "max_tokens": n_predict,
        "prompt": prompt,
        "stream": True,
        "oai_compat": False,
    })
    content = ""
    for data in res:
        if data["stop"]:
            assert data["timings"]["prompt_n"] == n_prompt
            assert data["timings"]["predicted_n"] == n_predicted
            assert match_regex(re_content, content)
        else:
            content += data["content"]


@pytest.mark.parametrize("n_slots", [1, 2])
def test_consistent_result_same_seed(n_slots: int):
    global server
    server.n_slots = n_slots
    server.start()
    last_res = None
    for _ in range(4):
        res = server.make_request("POST", "/completion", data={
            "prompt": "I believe the meaning of life is",
            "seed": 42,
            "temperature": 1.0,
            "cache_prompt": False,  # TODO: remove this once test_cache_vs_nocache_prompt is fixed
            "oai_compat": False,
        })
        if last_res is not None:
            assert res.body["content"] == last_res.body["content"]
        last_res = res


@pytest.mark.parametrize("n_slots", [1, 2])
def test_different_result_different_seed(n_slots: int):
    global server
    server.n_slots = n_slots
    server.start()
    last_res = None
    for seed in range(4):
        res = server.make_request("POST", "/completion", data={
            "prompt": "I believe the meaning of life is",
            "seed": seed,
            "temperature": 1.0,
            "cache_prompt": False,  # TODO: remove this once test_cache_vs_nocache_prompt is fixed
            "oai_compat": False,
        })
        if last_res is not None:
            assert res.body["content"] != last_res.body["content"]
        last_res = res


@pytest.mark.parametrize("n_batch", [16, 32])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_consistent_result_different_batch_size(n_batch: int, temperature: float):
    global server
    server.n_batch = n_batch
    server.start()
    last_res = None
    for _ in range(4):
        res = server.make_request("POST", "/completion", data={
            "prompt": "I believe the meaning of life is",
            "seed": 42,
            "temperature": temperature,
            "cache_prompt": False,  # TODO: remove this once test_cache_vs_nocache_prompt is fixed
            "oai_compat": False,
        })
        if last_res is not None:
            assert res.body["content"] == last_res.body["content"]
        last_res = res


@pytest.mark.skip(reason="This test fails on linux, need to be fixed")
def test_cache_vs_nocache_prompt():
    global server
    server.start()
    res_cache = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "seed": 42,
        "temperature": 1.0,
        "cache_prompt": True,
        "oai_compat": False,
    })
    res_no_cache = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "seed": 42,
        "temperature": 1.0,
        "cache_prompt": False,
        "oai_compat": False,
    })
    assert res_cache.body["content"] == res_no_cache.body["content"]


def test_completion_with_tokens_input():
    global server
    server.temperature = 0.0
    server.start()
    prompt_str = "I believe the meaning of life is"
    res = server.make_request("POST", "/tokenize", data={
        "content": prompt_str,
        "add_special": True,
    })
    assert res.status_code == 200
    tokens = res.body["tokens"]

    # single completion
    res = server.make_request("POST", "/completion", data={
        "prompt": tokens,
        "oai_compat": False,
    })
    assert res.status_code == 200
    assert type(res.body["content"]) == str

    # batch completion
    res = server.make_request("POST", "/completion", data={
        "prompt": [tokens, tokens],
        "oai_compat": False,
    })
    assert res.status_code == 200
    assert type(res.body) == list
    assert len(res.body) == 2
    assert res.body[0]["content"] == res.body[1]["content"]

    # mixed string and tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": [tokens, prompt_str],
        "oai_compat": False,
    })
    assert res.status_code == 200
    assert type(res.body) == list
    assert len(res.body) == 2
    assert res.body[0]["content"] == res.body[1]["content"]

    # mixed string and tokens in one sequence
    res = server.make_request("POST", "/completion", data={
        "prompt": [1, 2, 3, 4, 5, 6, prompt_str, 7, 8, 9, 10, prompt_str],
        "oai_compat": False,
    })
    assert res.status_code == 200
    assert type(res.body["content"]) == str


@pytest.mark.parametrize("n_slots,n_requests", [
    (1, 3),
    (2, 2),
    (2, 4),
    (4, 2), # some slots must be idle
    (4, 6),
])
def test_completion_parallel_slots(n_slots: int, n_requests: int):
    global server
    server.n_slots = n_slots
    server.temperature = 0.0
    server.start()

    PROMPTS = [
        ("Write a very long book.", "(very|special|big)+"),
        ("Write another a poem.", "(small|house)+"),
        ("What is LLM?", "(Dad|said)+"),
        ("The sky is blue and I love it.", "(climb|leaf)+"),
        ("Write another very long music lyrics.", "(friends|step|sky)+"),
        ("Write a very long joke.", "(cat|Whiskers)+"),
    ]
    def check_slots_status():
        should_all_slots_busy = n_requests >= n_slots
        time.sleep(0.1)
        res = server.make_request("GET", "/slots")
        n_busy = sum([1 for slot in res.body if slot["is_processing"]])
        if should_all_slots_busy:
            assert n_busy == n_slots
        else:
            assert n_busy <= n_slots

    tasks = []
    for i in range(n_requests):
        prompt, re_content = PROMPTS[i % len(PROMPTS)]
        tasks.append((server.make_request, ("POST", "/completion", {
            "prompt": prompt,
            "seed": 42,
            "temperature": 1.0,
            "oai_compat": False,
        })))
    tasks.append((check_slots_status, ()))
    results = parallel_function_calls(tasks)

    # check results
    for i in range(n_requests):
        prompt, re_content = PROMPTS[i % len(PROMPTS)]
        res = results[i]
        assert res.status_code == 200
        assert type(res.body["content"]) == str
        assert len(res.body["content"]) > 10
        # FIXME: the result is not deterministic when using other slot than slot 0
        # assert match_regex(re_content, res.body["content"])

# OpenAI legacy completion endpoint tests
@pytest.mark.parametrize("prompt,n_predict,expected_text,n_prompt,n_predicted", [
    ("I believe the meaning of life is", 8, "going to bed", 18, 8),
    ("Write a joke about", 16, "Why did the AI", 14, 16),
])
def test_completion_openai(prompt: str, n_predict: int, expected_text: str, n_prompt: int, n_predicted: int):
    global server
    server.start()
    
    # Test non-streaming response
    res = server.make_request("POST", "/completions", data={
        "model": "local-model",
        "prompt": prompt,
        "max_tokens": n_predict,
        "logprobs": 3,
        "echo": True
    })
    
    assert res.status_code == 200
    assert res.body["object"] == "text_completion"
    assert isinstance(res.body["id"], str)
    assert isinstance(res.body["created"], int)
    assert res.body["model"] == "local-model"
    
    # Check choices array
    assert len(res.body["choices"]) == 1
    choice = res.body["choices"][0]
    assert choice["index"] == 0
    assert isinstance(choice["text"], str)
    assert choice["finish_reason"] in ["stop", "length"]
    
    # Check logprobs
    assert choice["logprobs"] is not None
    assert "tokens" in choice["logprobs"]
    assert "token_logprobs" in choice["logprobs"]
    assert "top_logprobs" in choice["logprobs"]
    assert len(choice["logprobs"]["top_logprobs"]) == len(choice["logprobs"]["tokens"])
    
    # Check usage statistics
    assert "usage" in res.body
    assert res.body["usage"]["prompt_tokens"] == n_prompt
    assert res.body["usage"]["completion_tokens"] == n_predicted
    assert res.body["usage"]["total_tokens"] == n_prompt + n_predicted

@pytest.mark.parametrize("prompt,n_predict,expected_text,n_prompt,n_predicted", [
    ("I believe the meaning of life is", 8, "going to bed", 18, 8),
    ("Write a joke about", 16, "Why did the AI", 14, 16),
])
def test_completion_openai_stream(prompt: str, n_predict: int, expected_text: str, n_prompt: int, n_predicted: int):
    global server
    server.start()
    
    res = server.make_stream_request("POST", "/v1/completions", data={
        "prompt": prompt,
        "max_tokens": n_predict,
        "stream": True,
    })
    
    collected_text = ""
    is_first_chunk = True
    for data in res:
        assert "id" in data
        assert data["object"] == "text_completion"
        assert isinstance(data["created"], int)
        
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert isinstance(choice["text"], str)
        collected_text += choice["text"]
        
        if is_first_chunk:
            # First chunk should have model info
            is_first_chunk = False
        
        if choice["finish_reason"] is not None:
            # This is the last chunk
            assert choice["finish_reason"] in ["stop", "length"]
            assert "usage" in data
            assert data["usage"]["prompt_tokens"] == n_prompt
            assert data["usage"]["completion_tokens"] == n_predicted
            assert data["usage"]["total_tokens"] == n_prompt + n_predicted

@pytest.mark.parametrize("prompt,n_predict,expected_text,n_prompt,n_predicted", [
    ("I believe the meaning of life is", 8, "going to bed", 18, 8),
    ("Write a joke about", 16, "Why did the AI", 14, 16),
])
def test_completion_openai_no_logprobs(prompt: str, n_predict: int, expected_text: str, n_prompt: int, n_predicted: int):
    global server
    server.start()
    
    # Test non-streaming response
    res = server.make_request("POST", "/completions", data={
        "prompt": prompt,
        "max_tokens": n_predict,
        "echo": True
    })
    
    assert res.status_code == 200
    assert res.body["object"] == "text_completion"
    assert isinstance(res.body["id"], str)
    assert isinstance(res.body["created"], int)
    
    # Check choices array
    assert len(res.body["choices"]) == 1
    choice = res.body["choices"][0]
    assert choice["index"] == 0
    assert isinstance(choice["text"], str)
    assert choice["finish_reason"] in ["stop", "length"]
    
    # Verify logprobs is None when not requested
    assert choice["logprobs"] is None
    
    # Check usage statistics
    assert "usage" in res.body
    assert res.body["usage"]["prompt_tokens"] == n_prompt
    assert res.body["usage"]["completion_tokens"] == n_predicted
    assert res.body["usage"]["total_tokens"] == n_prompt + n_predicted