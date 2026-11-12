"""version3: Momentum Contrast (MoCo), many negatives without huge batches.

What is NEW compared to version2
--------------------------------
In version2 the negatives came from the current batch, so the number of
negatives was tied to the batch size (and to GPU memory). MoCo (He et al.,
2020) decouples the two with two ideas:

1. A *queue* (class History) of key embeddings from the last few batches.
   Each query is contrasted with its positive key and with every key in the
   queue, so we get `history * batch_size` negatives for the cost of one
   batch. The oldest keys are dropped as new ones are pushed.

2. A *momentum encoder* that produces the keys. Keys in the queue were
   computed several steps ago; if the key encoder changed quickly they would
   be inconsistent with each other. So the key encoder is not trained by
   gradients; it slowly follows the query encoder:

       theta_k <- m * theta_k + (1 - m) * theta_q,    m = 0.999

The loss is the same InfoNCE as before, with the positive in column 0:

    logits = [q . k_+ , q . queue] / tau,     target = 0

Here q = f_q(view 1), k_+ = f_k(view 2) (two augmentations of one image, as
in version2), all L2-normalised, tau = 0.07. The trained key encoder is saved
to checkpoints/moco_encoder.pth for version4.

Run:  python version3.py            (quick: python version3.py --epochs 1 --max-batches 50)
"""
import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from utils import Encoder, add_common_args, evaluate, get_data, get_dataloader, \
    get_device


class History:
    """FIFO queue of the most recent `history` key embeddings (a ring buffer)."""
    def __init__(self, history: int, dim: int, device):
        self.history = history
        self.buffer = torch.zeros(history, dim, device=device)
        self.ptr = 0
        self.count = 0

    @torch.no_grad()
    def push(self, x: torch.Tensor) -> None:
        idx = (self.ptr + torch.arange(len(x), device=x.device)) % self.history
        self.buffer[idx] = x
        self.ptr = (self.ptr + len(x)) % self.history
        self.count = min(self.count + len(x), self.history)

    def get(self) -> torch.Tensor:
        if self.count < self.history:  # wait until the queue is full
            return None
        return self.buffer


def train_moco(trainloader, encoder_q: nn.Module, encoder_k: nn.Module,
               optimizer: optim.Optimizer, buffer: History, device,
               m: float = 0.999, tau: float = 0.07, max_batches=None) -> None:
    encoder_q.train()
    for batch_idx, ((x_q, x_k), _) in enumerate(tqdm(trainloader)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x_q, x_k = x_q.to(device), x_k.to(device)

        with torch.no_grad():  # keys: no gradient, momentum encoder
            k = F.normalize(encoder_k(x_k), dim=1)  # B x d

        negatives = buffer.get()  # H x d, keys of earlier batches
        if negatives is not None:
            q = F.normalize(encoder_q(x_q), dim=1)  # B x d
            sim_p = (q * k).sum(dim=1, keepdim=True)  # B x 1
            sim_n = q @ negatives.T  # B x H
            logits = torch.cat([sim_p, sim_n], dim=1) / tau  # B x (H + 1)
            targets = torch.zeros(len(q), dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # momentum update of the key encoder
            with torch.no_grad():
                for param_k, param_q in zip(encoder_k.parameters(),
                                            encoder_q.parameters()):
                    param_k.data.mul_(m).add_(param_q.data, alpha=1. - m)

        # enqueue the newest keys (after using the queue, so a key is never
        # its own negative)
        buffer.push(k)
    if negatives is not None:
        print(f'Last batch loss: {loss.item():.4f}')


def main(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    trainset, _ = get_data(args.data, two_views=True)

    encoder_q = Encoder(latent_dim=args.latent_dim).to(device)
    encoder_k = copy.deepcopy(encoder_q)  # both start from the same weights
    for p in encoder_k.parameters():
        p.requires_grad = False
    optimizer = optim.Adam(encoder_q.parameters(), lr=args.lr)
    buffer = History(args.history * args.batch_size, args.latent_dim, device)

    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}')
        trainloader = get_dataloader(trainset, args.batch_size,
                                     num_workers=args.workers)
        train_moco(trainloader, encoder_q, encoder_k, optimizer, buffer, device,
                   args.m, args.tau, args.max_batches)
        torch.save(encoder_k.state_dict(), args.save)
        if args.test_every and (epoch + 1) % args.test_every == 0:
            evaluate(encoder_k, args, f'version3_epoch{epoch + 1}', device)
    print(f'Saved the momentum (key) encoder to {args.save}')
    evaluate(encoder_k, args, 'version3', device)


if __name__ == '__main__':
    parser = add_common_args(argparse.ArgumentParser(description=__doc__.split('\n')[0]))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--max-batches', type=int, default=None,
                        help='cap batches per epoch (for quick runs)')
    parser.add_argument('--history', type=int, default=10,
                        help='queue length, in batches')
    parser.add_argument('--m', type=float, default=0.999, help='momentum')
    parser.add_argument('--tau', type=float, default=0.07, help='temperature')
    parser.add_argument('--test-every', type=int, default=0,
                        help='run the linear probe every N epochs (0 = only at the end)')
    parser.add_argument('--save', default='./checkpoints/moco_encoder.pth')
    main(parser.parse_args())
