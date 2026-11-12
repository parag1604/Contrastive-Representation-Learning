"""version1: from one negative to many, the InfoNCE loss.

What is NEW compared to version0
--------------------------------
version0 compared each anchor with exactly one negative through a sigmoid.
Here every anchor x_i competes against *all* positives in the batch:

    logits_ij = f(x_i)^T f(y_j) / tau          (= -E(x_i, y_j))
    L_i       = -log softmax_j(logits_ij)[j = i]
              =  E(x_i, y_i) + log sum_j exp(-E(x_i, y_j))

This is the InfoNCE loss: the negative log-likelihood of a conditional
energy-based model p(y | x) whose intractable normalizer is estimated with
the other samples of the batch, the *in-batch negatives*. Its gradient pushes
the energy of the positive pair down and the energy of every negative up,
weighted by how plausible the model currently finds it (hard negatives get
the biggest push). See "Contrastive losses come from energy" in
https://parag1604.github.io/notes/Machine-Learning/Energy-Based-Models

Two further details that come with the softmax:
- embeddings are L2-normalised, so the score is a cosine similarity in [-1, 1]
  and cannot be inflated just by growing the norm of the embedding;
- the temperature tau sets how peaked the softmax is (small tau = focus on
  the hardest negatives).

Positives are still chosen with labels (another image of the same digit).
Because of that, some in-batch "negatives" have the same digit as the
anchor; we mask them out of the softmax so they are not pushed away.

Run:  python version1.py            (quick: python version1.py --iters 200)
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import Encoder, add_common_args, evaluate, get_data, get_device, \
    get_label_index


def get_contrastive_batch(trainset, labels, batch_size=256):
    """Anchors, one same-class positive per anchor, and the anchor labels."""
    idxs = torch.randint(0, len(trainset), (batch_size,))
    anchor, anchor_label, positive = [], [], []
    for i in idxs.tolist():
        x, y = trainset[i]
        anchor.append(x)
        anchor_label.append(y)
        j = np.random.choice(labels[y])
        while j == i:  # a different image of the same digit
            j = np.random.choice(labels[y])
        positive.append(trainset[j][0])
    return torch.stack(anchor), torch.stack(positive), torch.tensor(anchor_label)


def info_nce(anchor_embed, positive_embed, y, tau):
    a = F.normalize(anchor_embed, dim=1)
    p = F.normalize(positive_embed, dim=1)
    logits = a @ p.T / tau  # B x B, the diagonal holds the positive pairs
    # same digit but not the paired positive: not a true negative, drop it
    same_class = (y[:, None] == y[None, :]) & ~torch.eye(len(y), dtype=torch.bool,
                                                          device=y.device)
    logits = logits.masked_fill(same_class, float('-inf'))
    targets = torch.arange(len(y), device=logits.device)
    return F.cross_entropy(logits, targets)


def contrastive_train(encoder, trainset, optimizer, labels, device,
                      num_iters=1000, batch_size=256, tau=0.1):
    encoder.train()
    for i in range(1, num_iters + 1):
        anchor, positive, y = get_contrastive_batch(trainset, labels, batch_size)
        x_embed = encoder(torch.cat([anchor, positive]).to(device))
        anchor_embed, positive_embed = torch.split(x_embed, anchor.shape[0])
        loss = info_nce(anchor_embed, positive_embed, y.to(device), tau)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f'Iter: {i}, Loss: {loss.item():.4f}')
    encoder.eval()


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    trainset, _ = get_data(args.data)
    labels = get_label_index(trainset)

    encoder = Encoder(latent_dim=args.latent_dim).to(device)
    optimizer = optim.AdamW(encoder.parameters(), lr=args.lr)
    contrastive_train(encoder, trainset, optimizer, labels, device,
                      num_iters=args.iters, batch_size=args.batch_size,
                      tau=args.tau)
    evaluate(encoder, args, 'version1', device)


if __name__ == '__main__':
    parser = add_common_args(argparse.ArgumentParser(description=__doc__.split('\n')[0]))
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--tau', type=float, default=0.1, help='temperature')
    parser.set_defaults(batch_size=128, lr=5e-4)
    main(parser.parse_args())
