import re

import pytest
from utils import *

# We use a F16 MOE gguf as main model, and q4_0 as draft model

server = ServerPreset.stories15m_moe()

MODEL_DRAFT_FILE_URL = "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M-q4_0.gguf"

def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    # set default values
    server.model_draft = download_file(MODEL_DRAFT_FILE_URL)
    server.spec_type = "draft-simple"
    server.spec_draft_n_min = 4
    server.spec_draft_n_max = 8
    server.fa = "off"


@pytest.fixture(autouse=True)
def fixture_create_server():
    return create_server()


def test_with_and_without_draft():
    global server
    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.2,
        "top_k": 5,
        "seed": 4242,
        "n_predict": 16,
        "return_tokens": True,
    }

    server.model_draft = None  # disable draft model
    server.spec_type = None
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    tokens_no_draft = res.body["tokens"]
    server.stop()

    # create new server with draft model
    create_server()
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    tokens_draft = res.body["tokens"]

    assert tokens_no_draft == tokens_draft

    server.stop()
    create_server()
    assert server.spec_draft_n_max is not None
    server.spec_synth_rates = [0.0] * server.spec_draft_n_max
    server.start()
    res = server.make_request("POST", "/completion", data=request)

    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    assert res.body["timings"]["draft_n_accepted"] == 0
    assert res.body["tokens"] == tokens_no_draft


def test_different_draft_min_draft_max():
    global server
    test_values = [
        (1, 2),
        (1, 4),
        (4, 8),
        (4, 12),
        (8, 16),
    ]
    last_content = None
    for draft_min, draft_max in test_values:
        server.stop()
        server.spec_draft_n_min = draft_min
        server.spec_draft_n_max = draft_max
        server.start()
        res = server.make_request("POST", "/completion", data={
            "prompt": "I believe the meaning of life is",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 16,
        })
        assert res.status_code == 200
        if last_content is not None:
            assert last_content == res.body["content"]
        last_content = res.body["content"]


def test_synth_is_deterministic():
    global server
    assert server.spec_draft_n_max is not None
    server.spec_synth_rates = [0.75 ** (i + 1) for i in range(server.spec_draft_n_max)]
    server.start()

    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.2,
        "top_k": 5,
        "seed": 4242,
        "n_predict": 32,
    }
    responses = [server.make_request("POST", "/completion", data=request) for _ in range(2)]

    for res in responses:
        assert res.status_code == 200
        assert res.body["timings"]["draft_n"] > 0
    assert responses[0].body["timings"]["draft_n"] == responses[1].body["timings"]["draft_n"]
    assert responses[0].body["timings"]["draft_n_accepted"] == responses[1].body["timings"]["draft_n_accepted"]


def test_synth_ignores_target_tokens():
    global server
    assert server.spec_draft_n_max is not None
    server.spec_synth_rates = [1.0] * server.spec_draft_n_max
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "seed": 4242,
        "n_predict": 32,
    })

    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    assert res.body["timings"]["draft_n_accepted"] == res.body["timings"]["draft_n"]

    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "seed": 4242,
        "n_predict": 6,
        "grammar": 'root ::= "a"{5,5}',
    })
    assert res.status_code == 200, res.body

    res = server.make_request("POST", "/completion", data={
        "prompt": "Respond with only: OK",
        "temperature": 0.0,
        "seed": 4242,
        "n_predict": 64,
        "ignore_eos": True,
    })
    assert res.status_code == 200, res.body
    assert res.body["tokens_predicted"] == 64
    assert res.body["stop_type"] == "limit"


def test_draft_n_ubatch_smaller_than_target():
    global server
    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 16,
    }

    # draft with the default ubatch size of the draft context
    server.start()
    res_default = server.make_request("POST", "/completion", data=request)
    server.stop()
    assert res_default.status_code == 200
    assert res_default.body["timings"]["draft_n"] > 0

    # a smaller draft ubatch splits the draft batches into more ubatches, but must not change the result
    server.spec_draft_n_ubatch = 8
    server.start()
    res_small = server.make_request("POST", "/completion", data=request)
    server.stop()
    assert res_small.status_code == 200
    assert res_small.body["timings"]["draft_n"] > 0
    assert res_small.body["tokens"] == res_default.body["tokens"]


def test_slot_ctx_not_exceeded():
    global server
    server.n_ctx = 256
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello " * 248,
        "temperature": 0.0,
        "top_k": 1,
        "speculative.p_min": 0.0,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0


def test_with_ctx_shift():
    global server
    server.n_ctx = 256
    server.enable_ctx_shift = True
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello " * 248,
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 256,
        "speculative.p_min": 0.0,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0
    assert res.body["tokens_predicted"] == 256
    assert res.body["truncated"] == True


@pytest.mark.parametrize("n_branches", [2, 3, 4])
def test_tree_same_as_no_draft(n_branches: int, tmp_path):
    global server
    requests = [
        {
            "prompt": "I believe the meaning of life is",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 48,
            "return_tokens": True,
        },
        {
            "prompt": "I believe the meaning of life is",
            "temperature": 0.2,
            "top_k": 5,
            "seed": 4242,
            "n_predict": 48,
            "return_tokens": True,
        },
    ]

    server.model_draft = None
    server.spec_type = None
    server.start()
    tokens_no_draft = []
    for request in requests:
        res = server.make_request("POST", "/completion", data=request)
        assert res.status_code == 200
        tokens_no_draft.append(res.body["tokens"])
    server.stop()

    # a low split probability makes the drafter branch often
    create_server()
    server.spec_tree_branches = n_branches
    server.spec_draft_p_split = 0.01
    server.debug = True
    server.log_path = str(tmp_path / "server.log")
    server.start()
    for request, tokens in zip(requests, tokens_no_draft):
        res = server.make_request("POST", "/completion", data=request)
        assert res.status_code == 200
        assert res.body["timings"]["draft_n"] > 0
        assert res.body["tokens"] == tokens
    server.stop()

    with open(server.log_path, encoding="utf-8", errors="replace") as f:
        log = f.read()
    branches = [int(m) for m in re.findall(r"#branches=(\d+)", log)]
    assert len(branches) > 0
    assert max(branches) > 1
    assert max(branches) <= n_branches
    # some rounds accept a path that is not the first branch
    accepted = [int(m) for m in re.findall(r"accepted branch = (\d+)", log)]
    assert max(accepted) > 0


def test_tree_ngram_chain():
    global server
    request = {
        "prompt": "one two three four five six seven eight nine ten. " * 6,
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 96,
        "return_tokens": True,
    }

    server.model_draft = None
    server.spec_type = None
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    tokens_no_draft = res.body["tokens"]
    server.stop()

    # n-gram drafters cannot branch, their drafts are verified as single-branch trees
    create_server()
    server.model_draft = None
    server.spec_type = "ngram-simple"
    server.spec_tree_branches = 2
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    assert res.body["tokens"] == tokens_no_draft


def test_tree_backend_sampling_falls_back_to_cpu(tmp_path):
    global server
    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "n_predict": 32,
        "return_tokens": True,
    }

    server.model_draft = None
    server.spec_type = None
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    tokens_no_draft = res.body["tokens"]
    server.stop()

    create_server()
    server.spec_tree_branches = 3
    server.spec_draft_p_split = 0.01
    server.backend_sampling = True
    server.log_path = str(tmp_path / "server.log")
    server.start()
    res = server.make_request("POST", "/completion", data={**request, "backend_sampling": True})
    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    assert res.body["tokens"] == tokens_no_draft
    server.stop()

    with open(server.log_path, encoding="utf-8", errors="replace") as f:
        assert "backend sampling is not supported with --spec-tree-branches" in f.read()


def test_tree_ctx_limit_and_shift():
    global server
    server.n_ctx = 256
    server.enable_ctx_shift = True
    server.spec_tree_branches = 3
    server.spec_draft_p_split = 0.01
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello " * 248,
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 256,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0
    assert res.body["tokens_predicted"] == 256
    assert res.body["truncated"] == True


def test_tree_multi_slots():
    global server
    # short outputs that stay clear of near-tied tokens, other batch shapes can flip those
    requests = [{
        "prompt": prompt,
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 20,
        "return_tokens": True,
    } for prompt in [
        "I believe the meaning of life is",
        "The quick brown fox",
        "In the forest the owl said",
        "I believe the meaning of life is",
    ]]

    server.n_slots = 2
    server.model_draft = None
    server.spec_type = None
    server.start()
    tokens_no_draft = []
    for request in requests:
        res = server.make_request("POST", "/completion", data=request)
        assert res.status_code == 200
        tokens_no_draft.append(res.body["tokens"])
    server.stop()

    # the slots draft and verify their trees in the same batches
    create_server()
    server.n_slots = 2
    server.spec_tree_branches = 3
    server.spec_draft_p_split = 0.01
    server.start()
    results = parallel_function_calls([(server.make_request, ("POST", "/completion", request)) for request in requests])
    for res, tokens in zip(results, tokens_no_draft):
        assert res.status_code == 200
        assert res.body["timings"]["draft_n"] > 0
        assert res.body["tokens"] == tokens


@pytest.mark.parametrize("config,error", [
    ({"n_slots": 128},                                 "requires --parallel * --spec-tree-branches <= 256"),
    ({"model_draft": None, "spec_type": None},         "requires --spec-type"),
    ({"spec_type": "draft-eagle3"},                    "does not support --spec-type draft-eagle3"),
    ({"spec_draft_sampling": "probabilistic"},         "does not support --spec-draft-sampling probabilistic"),
    ({"spec_synth_rates": [0.5] * 8},                  "does not support synthetic acceptance"),
    ({"spec_draft_n_max": 2, "spec_draft_n_min": 1},   "requires --spec-draft-n-max >= --spec-tree-branches"),
])
def test_tree_unsupported_config(config: dict, error: str, tmp_path):
    global server
    server.spec_tree_branches = 3
    for key, value in config.items():
        setattr(server, key, value)
    server.log_path = str(tmp_path / "server.log")
    with pytest.raises(RuntimeError):
        server.start()
    with open(server.log_path, encoding="utf-8", errors="replace") as f:
        assert error in f.read()


@pytest.mark.parametrize("n_slots,n_requests", [
    (1, 2),
    (2, 2),
])
def test_multi_requests_parallel(n_slots: int, n_requests: int):
    global server
    server.n_slots = n_slots
    server.start()
    tasks = []
    for _ in range(n_requests):
        tasks.append((server.make_request, ("POST", "/completion", {
            "prompt": "I believe the meaning of life is",
            "temperature": 0.0,
            "top_k": 1,
        })))
    results = parallel_function_calls(tasks)
    for res in results:
        assert res.status_code == 200
        assert match_regex("(wise|kind|owl|answer)+", res.body["content"])


@pytest.mark.parametrize("n_branches", [1, 3])
def test_draft_token_probs(n_branches: int):
    # tokens accepted from the draft must carry the target model's probabilities,
    # same as without speculative decoding (#24271)
    global server
    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 16,
        "n_probs": 4,
    }

    server.model_draft = None  # disable draft model
    server.spec_type = None
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    probs_no_draft = res.body["completion_probabilities"]
    server.stop()

    create_server()
    if n_branches > 1:
        server.spec_tree_branches = n_branches
        server.spec_draft_p_split = 0.01
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    probs_draft = res.body["completion_probabilities"]

    assert len(probs_draft) == len(probs_no_draft)
    for tok_draft, tok_no_draft in zip(probs_draft, probs_no_draft):
        assert tok_draft["id"] == tok_no_draft["id"]
        assert len(tok_draft["top_logprobs"]) == len(tok_no_draft["top_logprobs"])
        assert abs(tok_draft["logprob"] - tok_no_draft["logprob"]) < 0.25
    # real logprobs are not all exactly 0.0 (prob 1.0)
    assert any(tok["logprob"] < -1e-4 for tok in probs_draft)
