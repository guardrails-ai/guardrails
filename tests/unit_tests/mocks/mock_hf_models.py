import os


def make_mock_model_and_tokenizer():
    """Returns a tuple of HF AutoModelForCausalLM and AutoTokenizer."""
    import torch

    torch.set_num_threads(1)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Can regenerate the sample pipe with this:
    # pipeline(
    #     "text-generation",
    #     "hf-internal-testing/tiny-random-gpt2",
    # ).save_pretrained("...")

    savedir = os.path.join(
        os.path.abspath(os.path.normpath(os.path.dirname(__file__))), "tiny-random-gpt2"
    )

    model = AutoModelForCausalLM.from_pretrained(
        savedir,
        local_files_only=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        savedir,
        local_files_only=True,
    )

    return model, tokenizer


def make_mock_pipeline():
    from transformers import pipeline

    model, tokenizer = make_mock_model_and_tokenizer()

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=False,
        device_map="cpu",  # Force CPU to avoid multithreaded fighting.
    )
    return pipe
