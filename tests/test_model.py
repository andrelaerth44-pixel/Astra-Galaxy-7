import torch

from src.generate import generate
from src.model import AstraGalaxy7


def tiny_config():
    return {
        "vocab_size": 128,
        "hidden_size": 64,
        "num_layers": 2,
        "num_heads": 4,
        "intermediate_size": 256,
        "max_seq_len": 64,
        "rope_theta": 10000.0,
        "dropout": 0.0,
    }


def test_forward_produces_loss():
    model = AstraGalaxy7(tiny_config())
    ids = torch.randint(0, 128, (2, 16))
    _, loss = model(ids, ids)
    assert torch.isfinite(loss)


def test_generation_changes_sequence_length():
    model = AstraGalaxy7(tiny_config())
    prompt = torch.randint(0, 128, (1, 8))
    output = generate(model, prompt, max_new_tokens=5, temperature=1.0, top_k=10, top_p=0.95)
    assert output.shape == (1, 13)
