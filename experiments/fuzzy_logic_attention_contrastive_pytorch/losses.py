"""Task and shared-term supervised contrastive objectives."""

import torch
from torch.nn import functional as F


def positive_partner_counts(term_ids):
    shared = (term_ids[:, None, :, None] == term_ids[None, :, None, :]).any(dim=3).any(dim=2)
    eye = torch.eye(len(term_ids), dtype=torch.bool, device=term_ids.device)
    return shared & ~eye, (shared & ~eye).sum(dim=1)


def shared_term_contrastive(attention_code, term_ids, temperature=1.0):
    """Supervised contrastive loss where functions sharing any term are positives.

    L_i = -1/|P(i)| sum_{p in P(i)} log[
        exp(cos(z_i,z_p)/T) / sum_{a != i} exp(cos(z_i,z_a)/T)]
    """
    positives, num_positives = positive_partner_counts(term_ids)
    eye = torch.eye(len(term_ids), dtype=torch.bool, device=term_ids.device)

    z = F.normalize(attention_code, dim=1)
    similarity = z @ z.T / temperature
    similarity = similarity.masked_fill(eye, float("-inf"))
    log_probability = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    per_anchor = -(log_probability.masked_fill(~positives, 0.0).sum(dim=1)) / num_positives.clamp_min(1)
    valid = num_positives > 0
    loss = per_anchor[valid].mean() if valid.any() else attention_code.sum() * 0.0
    return loss, num_positives.float().mean()


def training_loss(model, batch, lambda_contrastive, temperature):
    if lambda_contrastive > 0:
        prediction, attentions = model(batch.x, return_attention=True)
        # Response-token self-attention score across heads: [batch, heads].
        code = attentions[-1][:, :, -1, -1]
        contrastive, average_positives = shared_term_contrastive(
            code, batch.term_ids, temperature
        )
    else:
        prediction = model(batch.x)
        contrastive = prediction.new_zeros(())
        _, num_positives = positive_partner_counts(batch.term_ids)
        average_positives = num_positives.float().mean()
    task = F.mse_loss(prediction[:, -1], batch.y)
    return task + lambda_contrastive * contrastive, task, contrastive, average_positives
