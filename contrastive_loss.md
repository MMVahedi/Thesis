# Shared-Term Contrastive Loss (InfoNCE)

Source: `compgen/losses/contrastive.py` — `shared_term_contrastive`.

## Setup

- Batch of `N` examples.
- `z_i` = L2-normalized representation of example `i` (a unit vector in `R^D`).
  Produced by `F.normalize(representations, dim=1)`.
- `P(i)` = set of indices `j ≠ i` whose example shares at least one term with `i`.
- `|P(i)|` = number of such positive partners.
- `T > 0` = temperature.

## The loss

For each anchor `i` (skipping anchors with `P(i)` empty):

$$
\mathcal{L}_i
= -\,\frac{1}{|P(i)|}
\sum_{p \in P(i)}
\log
\frac{\exp\!\big(\cos(z_i, z_p) / T\big)}
{\sum_{a \neq i} \exp\!\big(\cos(z_i, z_a) / T\big)}
$$

Overall loss = mean of `L_i` over anchors with `|P(i)| > 0`:

$$
\mathcal{L} = \frac{1}{|\{i : |P(i)| > 0\}|} \sum_{i:\,|P(i)|>0} \mathcal{L}_i
$$

where

$$
\cos(z_i, z_j) = \frac{z_i \cdot z_j}{\lVert z_i \rVert \, \lVert z_j \rVert} = z_i \cdot z_j
\quad\text{(already unit vectors)}
$$

## Meaning

- **Numerator**: similarity to one positive partner `p` — we want it high.
- **Denominator**: similarity to *every other* example `a ≠ i` — all of them are negatives.
- **log-softmax**: this is the log-probability that, among all non-self examples, the model would "pick" `p` as the match for `i`.
- Minimizing `L_i` = making each positive partner stand out from the full set of negatives.
- `T` controls sharpness: smaller `T` → sharper distribution.

## Code correspondence

| Formula piece | Code |
|---|---|
| `z_i = unit(rep_i)` | `z = F.normalize(representations, dim=1)` |
| `cos(z_i, z_j) / T` | `similarity = z @ z.T / temperature` |
| `a ≠ i` (drop self) | `similarity = similarity.masked_fill(eye, float("-inf"))` |
| `log [exp / Σ exp]` | `log_probability = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)` |
| `Σ_{p ∈ P(i)}` | `log_probability.masked_fill(~positives, 0.0).sum(dim=1)` |
| `-1/|P(i)| · Σ` (average over positives) | `per_anchor = -(...) / num_positives.clamp_min(1)` |
| skip empty `P(i)`, mean | `loss = per_anchor[num_positives > 0].mean()` |

## Numerically stable log-softmax

$$
\log \frac{e^{s_{ij}}}{\sum_a e^{s_{ia}}}
= s_{ij} - \log \sum_a e^{s_{ia}}
= s_{ij} - \operatorname{logsumexp}_a(s_{ia})
$$

The diagonal is set to `-inf` so `exp(-inf) = 0` removes the self term from the denominator automatically.

## Edge cases

- Anchors with no positive partners are skipped (the average is undefined).
- If **no** anchor has a positive partner, the loss is `0` but still a differentiable tensor (`representations.sum() * 0.0`), so `.backward()` works.
- Return value also includes the mean positive-partner count as a diagnostic (`avg_positive_partners`).
