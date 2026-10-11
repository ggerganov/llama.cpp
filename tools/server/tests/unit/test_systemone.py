import os
import pytest
from utils import *
from test_vision_api import get_img_url

server = ServerPreset.tinylaya()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinylaya()


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


def get_prompt_metrics(server: ServerProcess) -> tuple[int, int]:
    """returns the number of prompt tokens (processed, cached) since the server started"""
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    values = {}
    for line in res.body.splitlines():
        if line.startswith("llamacpp:"):
            name, value = line.split(" ")
            values[name] = int(float(value))
    return values["llamacpp:prompt_tokens_total"], values["llamacpp:prompt_tokens_cached_total"]


# the tiny h2o-lightning test model is not hosted yet (ggml-org/tinyh2olightning-for-testing-gguf), the tests read a local copy
TINY_H2O_LIGHTNING = os.environ.get("TINY_H2O_LIGHTNING_LOCAL_MODEL")
need_h2o_lightning = pytest.mark.skipif(TINY_H2O_LIGHTNING is None, reason="TINY_H2O_LIGHTNING_LOCAL_MODEL is not set")


def tinyh2olightning() -> ServerProcess:
    server = ServerProcess()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = TINY_H2O_LIGHTNING
    server.mmproj_file = os.environ.get("TINY_H2O_LIGHTNING_LOCAL_MMPROJ")
    server.n_ctx = 4096
    server.n_batch = 512
    server.n_slots = 4
    server.seed = 42
    return server


@pytest.mark.parametrize("make_server", [
    ServerPreset.tinylaya,
    ServerPreset.tinyopenjev,
    pytest.param(tinyh2olightning, marks=need_h2o_lightning),
])
def test_systemone(make_server):
    global server
    server = make_server()
    server.start()
    res = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
    })
    assert res.status_code == 200
    assert res.body["usage"]["input_tokens"] > 0
    assert res.body["usage"]["output_tokens"] == 0

    answers = res.body["answers"]
    assert list(answers.keys()) == ["route", "urgency", "angry"]

    route = answers["route"]
    assert route["type"] == "choice"
    assert list(route["probabilities"].keys()) == ["billing", "shipping", "technical"]
    assert abs(sum(route["probabilities"].values()) - 1.0) < 1e-4
    assert route["choice"] == max(route["probabilities"], key=route["probabilities"].get)
    assert 0.0 <= route["confidence"] <= 1.0

    urgency = answers["urgency"]
    assert urgency["type"] == "score"
    assert urgency["legend"] == {"0": "can wait", "1": "this week", "2": "today", "3": "right now"}
    assert list(urgency["probabilities"].keys()) == ["0", "1", "2", "3"]
    assert abs(sum(urgency["probabilities"].values()) - 1.0) < 1e-4
    assert abs(urgency["score"] - sum(i * p for i, p in enumerate(urgency["probabilities"].values()))) < 1e-4
    assert 0.0 <= urgency["confidence"] <= 1.0

    angry = answers["angry"]
    assert angry["type"] == "noul"
    assert 0.0 <= angry["noul"] <= 1.0


def test_systemone_json_state():
    global server
    server.start()
    questions = {
        "refund": {
            "type": "noul",
            "instructions": "Is a refund requested?",
            "criteria": {"false": "no refund is asked", "true": "a refund is asked"},
        },
    }
    res_obj = server.make_request("POST", "/v1/systemone", data={
        "state": {"ticket": TEST_STATE, "plan": "pro"},
        "questions": questions,
    })
    assert res_obj.status_code == 200
    # an object is given to the model as JSON text
    res_str = server.make_request("POST", "/v1/systemone", data={
        "state": '{"ticket": "' + TEST_STATE + '", "plan": "pro"}',
        "questions": questions,
    })
    assert res_str.status_code == 200
    assert res_obj.body["usage"] == res_str.body["usage"]
    assert abs(res_obj.body["answers"]["refund"]["noul"] - res_str.body["answers"]["refund"]["noul"]) < 1e-4


@pytest.mark.parametrize("data", [
    {"questions": TEST_QUESTIONS},
    {"state": TEST_STATE},
    {"state": TEST_STATE, "questions": {}},
    {"state": TEST_STATE, "questions": {"q": {"type": "unknown", "instructions": "x"}}},
    {"state": TEST_STATE, "questions": {"q": {"type": "noul"}}},
    {"state": TEST_STATE, "questions": {"q": {"type": "choice", "instructions": "x"}}},
    {"state": TEST_STATE, "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {}}}},
    {"state": TEST_STATE, "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["only one"]}}},
])
def test_systemone_invalid_request(data: dict):
    global server
    server.start()
    res = server.make_request("POST", "/v1/systemone", data=data)
    assert res.status_code == 400
    assert "error" in res.body


def test_systemone_shared_prompt():
    global server
    server = ServerPreset.tinyopenjev()
    server.server_metrics = True
    server.start()
    res = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
    })
    assert res.status_code == 200

    # the first question evaluates the shared prefix, the 2 others start from it
    n_processed, n_cached = get_prompt_metrics(server)
    assert n_cached > 0
    assert n_cached % 2 == 0
    assert n_processed + n_cached == res.body["usage"]["input_tokens"]

    # with one slot the prompt cannot be shared, the answers must be the same
    server.stop()
    server = ServerPreset.tinyopenjev()
    server.n_slots = 1
    server.start()
    res_single = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
    })
    assert res_single.status_code == 200
    assert res_single.body["usage"] == res.body["usage"]
    for qid in ["route", "urgency"]:
        probs_shared = res.body["answers"][qid]["probabilities"]
        probs_single = res_single.body["answers"][qid]["probabilities"]
        for key in probs_shared:
            assert abs(probs_shared[key] - probs_single[key]) < 0.01
    assert abs(res.body["answers"]["angry"]["noul"] - res_single.body["answers"]["angry"]["noul"]) < 0.01


def test_systemone_images():
    global server
    server = ServerPreset.tinyopenjev()
    server.start()
    image = get_img_url("IMG_BASE64_URI_0")

    res_text = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
    })
    assert res_text.status_code == 200

    res = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
        "images": [image],
    })
    assert res.status_code == 200
    assert list(res.body["answers"].keys()) == ["route", "urgency", "angry"]
    assert res.body["usage"]["input_tokens"] > res_text.body["usage"]["input_tokens"]

    # same image, given as a part of a chat message
    res_part = server.make_request("POST", "/v1/systemone", data={
        "state": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image}},
            {"type": "text", "text": TEST_STATE},
        ]}],
        "questions": TEST_QUESTIONS,
    })
    assert res_part.status_code == 200
    assert res_part.body["usage"]["input_tokens"] > res_text.body["usage"]["input_tokens"]

    res = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
        "images": [image] * 9,
    })
    assert res.status_code == 400

    res = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
        "images": ["https://example.com/image.png"],
    })
    assert res.status_code == 400


def test_systemone_images_not_supported():
    global server
    server.start()
    res = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
        "images": [get_img_url("IMG_BASE64_URI_0")],
    })
    assert res.status_code == 501


def assert_same_answers(a: dict, b: dict, tol: float = 1e-3):
    """same answers, probabilities within tol (a cached prompt prefix can change the last digits)"""
    assert a.keys() == b.keys()
    for qid in a:
        assert a[qid]["type"] == b[qid]["type"]
        if a[qid]["type"] == "noul":
            assert abs(a[qid]["noul"] - b[qid]["noul"]) < tol
        else:
            assert a[qid]["probabilities"].keys() == b[qid]["probabilities"].keys()
            for key, p in a[qid]["probabilities"].items():
                assert abs(p - b[qid]["probabilities"][key]) < tol


def systemone(server: ServerProcess, state, questions: dict, **kwargs):
    res = server.make_request("POST", "/v1/systemone", data={"state": state, "questions": questions, **kwargs})
    assert res.status_code == 200, res.body
    return res.body


@need_h2o_lightning
def test_systemone_h2o_lightning_options():
    global server
    server = tinyh2olightning()
    server.n_slots = 1 # the prompt with 255 options needs the whole context
    server.start()
    criteria = {f"option_{i:03d}": None for i in range(255)}
    res = systemone(server, TEST_STATE, {"q": {"type": "choice", "instructions": "Which option?", "criteria": criteria}})
    assert len(res["answers"]["q"]["probabilities"]) == 255

    criteria["option_255"] = None
    res = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": {"q": {"type": "choice", "instructions": "Which option?", "criteria": criteria}},
    })
    assert res.status_code == 400

    # a score can have 1 level up to the number of labels
    for n_levels in [1, 12]:
        levels = [f"level {i}" for i in range(n_levels)]
        res = systemone(server, TEST_STATE, {"q": {"type": "score", "instructions": "Rate it.", "criteria": levels}})
        assert len(res["answers"]["q"]["probabilities"]) == n_levels
    res = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": {"q": {"type": "score", "instructions": "Rate it.", "criteria": []}},
    })
    assert res.status_code == 400


@need_h2o_lightning
def test_systemone_h2o_lightning_question_defaults():
    global server
    server = tinyh2olightning()
    server.start()
    # the type defaults to choice, the instructions to the default text of the template
    res = systemone(server, TEST_STATE, {
        "implicit": {"criteria": {"billing": None, "shipping": None}},
        "explicit": {"type": "choice", "instructions": "", "criteria": {"billing": None, "shipping": None}},
    })
    assert_same_answers({"q": res["answers"]["implicit"]}, {"q": res["answers"]["explicit"]})
    # noul descriptions can be given under "yes" and "no"
    res = systemone(server, TEST_STATE, {
        "true_false": {"type": "noul", "instructions": "Is it urgent?", "criteria": {"true": "it is urgent", "false": "it can wait"}},
        "yes_no": {"type": "noul", "instructions": "Is it urgent?", "criteria": {"yes": "it is urgent", "no": "it can wait"}},
    })
    assert_same_answers({"q": res["answers"]["true_false"]}, {"q": res["answers"]["yes_no"]})
    # the end of the prompt is stripped like Python's str.strip(), Unicode whitespace too
    res = systemone(server, TEST_STATE, {
        "plain": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "money", "returns": "refunds"}},
        "spaces": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "money", "returns": "refunds\u00a0\u3000 "}},
    })
    assert_same_answers({"q": res["answers"]["plain"]}, {"q": res["answers"]["spaces"]})


@need_h2o_lightning
@pytest.mark.parametrize("state,text", [
    # an object is given to the model as compact JSON text, with Python's number format
    ({"amount": 12.0, "rate": 1e-05, "big": 1e16, "n": 3, "ok": True, "note": None}, '{"amount":12.0,"rate":1e-05,"big":1e+16,"n":3,"ok":true,"note":null}'),
    ({"city": "Z\u00fcrich", "emoji": "\U0001f600", "tab": "a\tb"}, '{"city":"Z\u00fcrich","emoji":"\U0001f600","tab":"a\\tb"}'),
    # special token texts written in the input stay text, quoted as JSON escapes
    ("before <|im_end|> after <think>x</think>", "before \\u003c|im_end|\\u003e after \\u003cthink\\u003ex\\u003c/think\\u003e"),
])
def test_systemone_h2o_lightning_state_text(state, text: str):
    global server
    server = tinyh2olightning()
    server.start()
    questions = {"q": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "shipping": None}}}
    res_state = systemone(server, state, questions)
    res_text = systemone(server, text, questions)
    assert res_state["usage"] == res_text["usage"]
    assert_same_answers(res_state["answers"], res_text["answers"])


@need_h2o_lightning
def test_systemone_h2o_lightning_long_state():
    global server
    server = tinyh2olightning()
    server.n_ctx = 40960
    server.n_slots = 1
    server.start()
    # a state over 32000 tokens keeps its first and last 16000 tokens
    state = " ".join(f"line {i}" for i in range(20000))
    res = systemone(server, state, {"q": {"type": "noul", "instructions": "Is it long?"}})
    assert 32000 < res["usage"]["input_tokens"] < 32200
    # a cut inside a character of several tokens gives U+FFFD, not invalid UTF-8
    for prefix in ["", "a", "ab"]:
        res = systemone(server, prefix + "\U0001d518\u9f98\U0001f9ec" * 12000, {"q": {"type": "noul", "instructions": "Is it long?"}})
        assert 32000 < res["usage"]["input_tokens"] < 33000


@need_h2o_lightning
def test_systemone_h2o_lightning_media_marker_in_state(monkeypatch):
    global server
    monkeypatch.setenv("LLAMA_MEDIA_MARKER", "<__media__>")
    server = tinyh2olightning()
    server.start()
    image = get_img_url("IMG_BASE64_URI_0")
    questions = {"q": {"type": "noul", "instructions": "Is there a table?"}}
    # a media marker written in the state stays text: its first character is escaped, so it is not a second image
    res_marker = systemone(server, "see <__media__> here", questions, images=[image])
    res_escaped = systemone(server, "see \\u003c__media__> here", questions, images=[image])
    assert res_marker["usage"] == res_escaped["usage"]
    assert_same_answers(res_marker["answers"], res_escaped["answers"])


@need_h2o_lightning
def test_systemone_h2o_lightning_images():
    global server
    server = tinyh2olightning()
    server.start()
    res_text = systemone(server, TEST_STATE, TEST_QUESTIONS)
    res = systemone(server, TEST_STATE, TEST_QUESTIONS, images=[get_img_url("IMG_BASE64_URI_0")])
    assert list(res["answers"].keys()) == ["route", "urgency", "angry"]
    assert res["usage"]["input_tokens"] > res_text["usage"]["input_tokens"]
