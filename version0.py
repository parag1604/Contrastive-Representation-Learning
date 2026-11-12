"""version0: learning an embedding from "same / different" pairs.

What this file teaches
----------------------
A representation is good if images of the same thing land close together and
images of different things land far apart. We make that literal:

- sample an *anchor* image, a *positive* (another image of the same digit)
  and a *negative* (an image of a different digit);
- score a pair by the dot product of its embeddings, s(x, y) = f(x)^T f(y),
  i.e. a negative energy E(x, y) = -s(x, y);
- train a binary classifier on the scores: sigmoid(s) should be 1 for the
  (anchor, positive) pair and 0 for the (anchor, negative) pair.

This is binary cross-entropy on similarities: push the energy of the positive
pair down and the energy of the negative pair up. Note that the labels are
still used to pick positives and negatives (a *supervised* contrastive
setup); later versions remove that.

Afterwards we look at the embedding with t-SNE and measure how linearly
separable the digits are with a linear probe (utils.evaluate).

Run:  python version0.py            (quick: python version0.py --iters 200)
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import Encoder, add_common_args, evaluate, get_data, get_device, \
    get_label_index


def get_contrastive_batch(trainset, labels, batch_size=64):
    """One positive and one negative for each of `batch_size` anchors."""
    idxs = torch.randint(0, len(trainset), (batch_size,))
    anchor = torch.stack([trainset[i][0] for i in idxs])
    anchor_label = [trainset[i][1] for i in idxs]
    positive, negative = [], []
    for label in anchor_label:
        positive.append(np.random.choice(labels[label]))
        neg_label = np.random.choice(10)
        while neg_label == label:
            neg_label = np.random.choice(10)
        negative.append(np.random.choice(labels[neg_label]))
    positive = torch.stack([trainset[i][0] for i in positive])
    negative = torch.stack([trainset[i][0] for i in negative])
    return anchor, positive, negative


def contrastive_train(encoder, trainset, optimizer, labels, device,
                      num_iters=1000, batch_size=64):
    encoder.train()
    for i in range(1, num_iters + 1):
        anchor, positive, negative = get_contrastive_batch(
            trainset, labels, batch_size)

        # one forward pass for all three groups, then split them again
        X = torch.cat([anchor, positive, negative]).to(device)
        x_embed = encoder(X)
        anchor_embed, positive_embed, negative_embed = torch.split(
            x_embed, split_size_or_sections=anchor.shape[0])

        # similarity scores act as logits of "is this pair a match?"
        pos_sim = torch.sum(anchor_embed * positive_embed, dim=1)
        neg_sim = torch.sum(anchor_embed * negative_embed, dim=1)
        logits = torch.cat([pos_sim, neg_sim])
        targets = torch.cat([torch.ones(anchor.shape[0]),
                             torch.zeros(anchor.shape[0])]).to(device)
        loss = F.binary_cross_entropy_with_logits(logits, targets)

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
                      num_iters=args.iters, batch_size=args.batch_size)
    evaluate(encoder, args, 'version0', device)


if __name__ == '__main__':
    parser = add_common_args(argparse.ArgumentParser(description=__doc__.split('\n')[0]))
    parser.add_argument('--iters', type=int, default=2000)
    parser.set_defaults(batch_size=64, lr=5e-4)
    main(parser.parse_args())
