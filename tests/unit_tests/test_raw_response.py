from guardrails.classes.llm.llm_response import LLMResponse
from guardrails.classes.validation_outcome import ValidationOutcome
from guardrails.classes.history import Call, Iteration, Outputs, CallInputs


class TestLLMResponseRawResponse:
    def test_raw_response_default_none(self):
        resp = LLMResponse(output="test")
        assert resp.raw_response is None

    def test_raw_response_set_and_get(self):
        raw = {
            "choices": [
                {
                    "logprobs": {
                        "content": [{"token": "hello", "logprob": -0.5}]
                    }
                }
            ]
        }
        resp = LLMResponse(output="test", raw_response=raw)
        assert resp.raw_response == raw
        logprobs = resp.raw_response["choices"][0]["logprobs"]
        assert logprobs["content"][0]["token"] == "hello"
        assert logprobs["content"][0]["logprob"] == -0.5

    def test_raw_response_serialization(self):
        raw = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}
        resp = LLMResponse(
            output="test",
            prompt_token_count=10,
            response_token_count=20,
            raw_response=raw,
        )
        dumped = resp.model_dump()
        assert dumped["raw_response"] == raw
        assert dumped["output"] == "test"

    def test_arbitrary_callable_no_raw_response(self):
        resp = LLMResponse(output="test")
        assert resp.raw_response is None


class TestValidationOutcomeRawResponse:
    def test_raw_response_default_none(self):
        vo = ValidationOutcome(
            callId="test-call",
            rawLlmOutput="test",
            validatedOutput="validated",
            validationPassed=True,
        )
        assert vo.raw_response is None

    def test_raw_response_set_and_get(self):
        raw = {
            "choices": [
                {
                    "logprobs": {
                        "content": [{"token": "hello", "logprob": -0.5}]
                    }
                }
            ]
        }
        vo = ValidationOutcome(
            callId="test-call",
            rawLlmOutput="test",
            validatedOutput="validated",
            validationPassed=True,
            rawResponse=raw,
        )
        assert vo.raw_response == raw
        logprobs = vo.raw_response["choices"][0]["logprobs"]
        assert logprobs["content"][0]["token"] == "hello"

    def test_raw_response_serialization(self):
        raw = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}
        vo = ValidationOutcome(
            callId="test-call",
            rawLlmOutput="test",
            validatedOutput="validated",
            validationPassed=True,
            rawResponse=raw,
        )
        dumped = vo.model_dump()
        assert dumped["raw_response"] == raw

    def test_logprobs_access_pattern(self):
        raw = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "content": "Hello world!",
                        "role": "assistant",
                    },
                    "logprobs": {
                        "content": [
                            {
                                "token": "Hello",
                                "logprob": -0.12,
                                "bytes": None,
                                "top_logprobs": [],
                            },
                            {
                                "token": " world",
                                "logprob": -0.05,
                                "bytes": None,
                                "top_logprobs": [],
                            },
                            {
                                "token": "!",
                                "logprob": -0.01,
                                "bytes": None,
                                "top_logprobs": [],
                            },
                        ]
                    },
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        vo = ValidationOutcome(
            callId="test-call",
            rawLlmOutput="Hello world!",
            validatedOutput="Hello world!",
            validationPassed=True,
            rawResponse=raw,
        )
        content_logprobs = vo.raw_response["choices"][0]["logprobs"]["content"]
        assert len(content_logprobs) == 3
        assert content_logprobs[0]["token"] == "Hello"
        assert content_logprobs[0]["logprob"] == -0.12
        assert content_logprobs[2]["token"] == "!"
        assert content_logprobs[2]["logprob"] == -0.01


class TestValidationOutcomeFromGuardHistory:
    def test_from_guard_history_with_raw_response(self):
        raw = {"choices": [{"logprobs": {"content": []}}]}

        llm_resp = LLMResponse(output="raw output", raw_response=raw)
        outputs = Outputs(llm_response_info=llm_resp)
        iteration = Iteration(callId="call-1", index=0, outputs=outputs)
        call = Call(inputs=CallInputs())
        call.iterations.push(iteration)

        vo = ValidationOutcome.from_guard_history(call)
        assert vo.raw_response == raw
        assert vo.raw_llm_output == "raw output"

    def test_from_guard_history_without_raw_response(self):
        llm_resp = LLMResponse(output="raw output")
        outputs = Outputs(llm_response_info=llm_resp)
        iteration = Iteration(callId="call-1", index=0, outputs=outputs)
        call = Call(inputs=CallInputs())
        call.iterations.push(iteration)

        vo = ValidationOutcome.from_guard_history(call)
        assert vo.raw_response is None
        assert vo.raw_llm_output == "raw output"

    def test_from_guard_history_no_llm_response(self):
        outputs = Outputs()
        iteration = Iteration(callId="call-1", index=0, outputs=outputs)
        call = Call(inputs=CallInputs())
        call.iterations.push(iteration)

        vo = ValidationOutcome.from_guard_history(call)
        assert vo.raw_response is None
        assert vo.raw_llm_output is None
