"""
The vLLM batch_decode_coeff probe's stream parsing and timing, tested without
a server.

The probe itself calls vLLM and is run by the owner on the GPU box. What can
be wrong offline is reading per-token time from the stream: counting the
usage chunk as a token, or timing prefill as decode, would each give a
plausible-looking wrong coefficient, so both are pinned here.
"""

import json
import os

import pytest

from probes import batch_decode_probe
from probes import vllm_batch_probe as vp
from probes.vllm_batch_probe import (DONE, default_out, infer_quantization,
                                     parse_sse_line, request_body, stream_timing)


def _data(chunk):
    return f"data: {json.dumps(chunk)}\n".encode()


def _token(text="1"):
    return {"object": "text_completion", "choices": [{"index": 0, "text": text,
                                                      "finish_reason": None}]}


USAGE = {"object": "text_completion", "choices": [],
         "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}}


def test_sse_lines_parse_to_chunks_done_or_nothing():
    assert parse_sse_line(_data(_token("7"))) == _token("7")
    assert parse_sse_line(b'data: {"choices": []}\r\n') == {"choices": []}
    assert parse_sse_line("data: [DONE]") is DONE
    for skip in (b"\n", b"\r\n", "", ": keep-alive\n"):
        assert parse_sse_line(skip) is None
    with pytest.raises(ValueError):
        parse_sse_line(b"data: {not json\n")


def test_per_token_time_is_the_gap_between_token_chunks():
    # Prefill takes 0.5 s, then a token every 30 ms. The usage chunk comes
    # after the last token and must not count as one.
    t0 = 100.0
    events = [(t0 + 0.5 + 0.03 * i, _token()) for i in range(4)]
    events.append((t0 + 0.5 + 0.03 * 3 + 0.001, USAGE))
    t = stream_timing(t0, events)
    assert t["per_token_s"] == pytest.approx(0.03)
    assert t["ttft_s"] == pytest.approx(0.5)
    assert t["chunks"] == 4
    assert t["completion_tokens"] == 4
    assert t["tokens_per_chunk"] == pytest.approx(1.0)


def test_coalesced_chunks_show_up_in_tokens_per_chunk():
    events = [(i * 0.06, _token("1\n2")) for i in range(4)]
    events.append((0.25, dict(USAGE, usage={"completion_tokens": 8})))
    assert stream_timing(0.0, events)["tokens_per_chunk"] == pytest.approx(2.0)


def test_without_usage_tokens_per_chunk_is_unknown():
    t = stream_timing(0.0, [(0.1, _token()), (0.2, _token())])
    assert t["completion_tokens"] is None and t["tokens_per_chunk"] is None


def test_one_token_chunk_cannot_be_timed():
    with pytest.raises(ValueError):
        stream_timing(0.0, [(0.1, _token()), (0.2, USAGE)])


def test_request_streams_reports_usage_and_defeats_the_prefix_cache():
    a, b = request_body("m", 256, 1234567890), request_body("m", 256, 9876543210)
    assert a["stream"] is True and a["stream_options"] == {"include_usage": True}
    assert a["max_tokens"] == 256 and a["ignore_eos"] is True
    assert a["prompt"].startswith("# run 1234567890\n")
    assert a["prompt"] != b["prompt"]


@pytest.mark.parametrize("model,quant", [
    ("Qwen/Qwen2.5-Coder-7B-Instruct-AWQ", "awq"),
    ("Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4", "gptq"),
    ("neuralmagic/Qwen2.5-7B-Instruct-FP8", "fp8"),
    ("Qwen/Qwen2.5-Coder-1.5B-Instruct", None),
])
def test_quantisation_is_read_from_the_model_name(model, quant):
    assert infer_quantization(model) == quant


def test_output_never_lands_on_the_ollama_fit():
    path = default_out("cuda-5060ti")
    assert path.endswith(os.path.join("cuda-5060ti", "vllm_batch_decode.json"))
    assert os.path.basename(path) != "batch_decode.json"


def test_the_fit_and_overlap_check_are_the_ollama_probes_own():
    assert vp.fit_coeff is batch_decode_probe.fit_coeff
    assert vp.overlap is batch_decode_probe.overlap
