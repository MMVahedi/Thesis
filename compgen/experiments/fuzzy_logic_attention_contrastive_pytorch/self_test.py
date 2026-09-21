"""Fast executable checks for the PyTorch port and notebook environment."""

import torch

from config import ExperimentConfig, TASKS
from data import FuzzyLogicGenerator
from losses import shared_term_contrastive, training_loss
from model import StandardAttentionTransformer, _relative_position_bucket


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expected_datasets = {
        "logic_3var_2term": (8, 8, 7, 1),
        "logic_4var_2term": (16, 33, 33, 6),
        "logic_4var_3term": (16, 110, 110, 4),
        "logic_5var_2term": (32, 138, 138, 28),
    }
    for task, task_spec in TASKS.items():
        config = ExperimentConfig(task=task)
        generator = FuzzyLogicGenerator(**task_spec, seed=config.seed)
        description = generator.describe()
        actual = (
            description["conjunctions"], description["train_functions"],
            description["test_functions"], description["ood_functions"],
        )
        assert actual == expected_datasets[task], (task, actual)
        batch = generator.sample(8, config.seq_len, "train", 0, device)
        assert batch.x.shape == (8, config.seq_len, task_spec["num_variables"] + 1)
        inputs = batch.x[:, :, :-1]
        literals = torch.where(
            batch.latents[:, None].bool(), inputs[:, :, None], 1.0 - inputs[:, :, None]
        )
        targets = literals.amin(dim=-1).amax(dim=-1, keepdim=True)
        assert torch.allclose(targets[:, :-1], batch.x[:, :-1, -1:], atol=1e-6)

    config = ExperimentConfig(task="logic_4var_2term")
    generator = FuzzyLogicGenerator(**config.task_spec, seed=config.seed)
    model = StandardAttentionTransformer(5, config).to(device)
    assert sum(parameter.numel() for parameter in model.parameters()) == 151_009
    batch = generator.sample(16, config.seq_len, "train", 1, device)
    model.eval()
    prediction, attentions = model(batch.x, return_attention=True)
    assert prediction.shape == (16, config.seq_len, 1)
    for attention in attentions:
        assert attention.shape == (16, 8, config.seq_len, config.seq_len)
        assert torch.allclose(
            attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1)), atol=1e-5
        )

    model.train()
    loss, task_loss, contrastive, positives = training_loss(model, batch, 0.05, 1.0)
    assert all(torch.isfinite(value) for value in (loss, task_loss, contrastive, positives))
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    codes = torch.randn(2, 8, device=device, requires_grad=True)
    disjoint_terms = torch.tensor([[0, 1], [2, 3]], device=device)
    edge_loss, edge_positives = shared_term_contrastive(codes, disjoint_terms, 1.0)
    assert edge_loss.item() == 0.0 and edge_positives.item() == 0.0
    edge_loss.backward()
    assert codes.grad is not None and torch.isfinite(codes.grad).all()

    relative = torch.arange(16, device=device)[None] - torch.arange(16, device=device)[:, None]
    buckets = _relative_position_bucket(relative, 16, 16)
    assert buckets.min().item() == 0 and buckets.max().item() == 15
    print(f"PyTorch self-test passed on {device}.")


if __name__ == "__main__":
    main()
