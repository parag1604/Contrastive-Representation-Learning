"""version2: throwing the labels away, SimCLR-style self-supervision.

What is NEW compared to version1
--------------------------------
version1 still needed labels to find a positive ("another 7"). Here the
positive is *another augmentation of the same image*: every image is randomly
rotated and cropped twice (utils.TwoViews), and the two views must be matched
against everything else in the batch. No label is used during training; the
labels appear only in the evaluation (linear probe and t-SNE colours).

With a batch of B images we get 2B views. For each view, its twin is the
positive and the other 2B - 2 views are in-batch negatives. The loss is
InfoNCE over this 2B x 2B similarity matrix (called NT-Xent in the SimCLR
paper, Chen et al. 2020), applied symmetrically to both views.

What the network learns is exactly what the augmentation leaves unchanged:
the identity of the digit, not its exact position or angle. Because the
negatives come from the batch, a bigger batch means more negatives and a
better estimate of the InfoNCE normalizer; version3 removes that coupling.

Run:  python version2.py            (quick: python version2.py --epochs 1 --max-batches 50)
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from utils import Encoder, add_common_args, evaluate, get_data, get_dataloader, \
    get_device


def nt_xent(z1, z2, tau):
    """InfoNCE over 2B views: the positive of view i is its twin."""
    B = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2]), dim=1)  # 2B x d
    logits = z @ z.T / tau  # 2B x 2B
    # a view is not its own negative: remove the diagonal from the softmax
    logits.fill_diagonal_(float('-inf'))
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)])
    return F.cross_entropy(logits, targets.to(z.device))


def train_simclr(encoder, trainloader, optimizer, device, tau=0.1,
                 max_batches=None):
    encoder.train()
    for batch_idx, ((x1, x2), _) in enumerate(tqdm(trainloader)):  # labels unused
        if max_batches is not None and batch_idx >= max_batches:
            break
        z = encoder(torch.cat([x1, x2]).to(device))
        z1, z2 = torch.split(z, x1.shape[0])
        loss = nt_xent(z1, z2, tau)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Last batch loss: {loss.item():.4f}')
    encoder.eval()


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    trainset, _ = get_data(args.data, two_views=True)

    encoder = Encoder(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}')
        trainloader = get_dataloader(trainset, args.batch_size,
                                     num_workers=args.workers)
        train_simclr(encoder, trainloader, optimizer, device, args.tau,
                     args.max_batches)
    evaluate(encoder, args, 'version2', device)


if __name__ == '__main__':
    parser = add_common_args(argparse.ArgumentParser(description=__doc__.split('\n')[0]))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max-batches', type=int, default=None,
                        help='cap batches per epoch (for quick runs)')
    parser.add_argument('--tau', type=float, default=0.1, help='temperature')
    parser.set_defaults(lr=1e-3)
    main(parser.parse_args())
