import os

import pytest
from utils import ServerProcess


@pytest.fixture(scope="session", autouse=True)
def load_server_presets():
    """This local checkpoint test does not need the downloaded model presets."""
    pass


TEST_STATE = "I was charged twice for my order last week and nobody has replied."

TEST_QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "payments and refunds", "shipping": None, "technical": None},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["can wait", "this week", "today", "right now"],
    },
    "angry": {
        "type": "noul",
        "instructions": "Is the customer angry?",
    },
}


def test_gliner_systemone_local():
    """Optional F32 GLiNER2.5-Decide regression; never downloads the checkpoint."""
    model = os.environ.get("GLINER_F32_LOCAL_MODEL")
    if not model:
        pytest.skip("GLINER_F32_LOCAL_MODEL is not set")
    server = ServerProcess()
    server.model_file = model
    server.model_hf_repo = None
    server.model_hf_file = None
    server.n_gpu = 0
    server.n_ctx = 512
    server.n_batch = 512
    server.n_ubatch = 512
    server.n_slots = 1
    server.start()

    question = {"type": "choice", "instructions": "", "criteria": {
        "refund_request": None, "order_status": None, "other": None,
    }}
    res = server.make_request("POST", "/v1/systemone", data={
        "state": "Please refund my order", "questions": {"intent": question},
    })
    assert res.status_code == 200
    answer = res.body["answers"]["intent"]
    assert answer["choice"] == "refund_request"
    assert answer["probabilities"]["refund_request"] == pytest.approx(0.9914846, abs=3e-4)
    assert res.body["usage"] == {"input_tokens": 22, "output_tokens": 0}

    res = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE, "questions": TEST_QUESTIONS,
    })
    assert res.status_code == 200
    assert list(res.body["answers"]) == list(TEST_QUESTIONS)
    assert sum(res.body["answers"]["route"]["probabilities"].values()) == pytest.approx(1.0)
    assert 0 <= res.body["answers"]["urgency"]["score"] <= 3
    assert 0 <= res.body["answers"]["angry"]["noul"] <= 1

    composed = server.make_request("POST", "/tokenize", data={"content": "caf\u00e9", "add_special": False})
    decomposed = server.make_request("POST", "/tokenize", data={"content": "cafe\u0301", "add_special": False})
    assert composed.status_code == decomposed.status_code == 200
    assert composed.body["tokens"] == decomposed.body["tokens"]

    invalid = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE, "questions": {"entities": {"type": "spans", "instructions": "Find people"}},
    })
    assert invalid.status_code == 400
