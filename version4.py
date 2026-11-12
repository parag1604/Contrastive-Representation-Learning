"""version4: what is pretraining worth? Fine-tuning with only 1000 labels.

What is NEW compared to version3
--------------------------------
So far we judged representations with a frozen encoder (linear probe). The
practical use of self-supervised pretraining is different: lots of unlabelled
data, few labels. Here we take only `--num-labels` (default 1000) labelled
training images and train the *whole* network (encoder + linear classifier),
starting from

- a random initialisation, and
- the MoCo encoder saved by version3 (checkpoints/moco_encoder.pth).

Epoch 0 trains only the linear classifier for one epoch (a linear probe on the
small labelled set); afterwards encoder and classifier are fine-tuned
together. The two test-accuracy curves are plotted to outputs/finetune.png:
the pretrained encoder should start far higher and converge faster.

Run:  python version3.py && python version4.py      (quick: --epochs 3)
"""
import argparse
import os

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import Classifier, Encoder, accuracy, get_data, get_dataloader, \
    get_device


def train(encoder, classifier, dataloader, optimizer, device):
    encoder.train()
    classifier.train()
    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)
        loss = F.nll_loss(classifier(encoder(x)), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def run(args, load: bool):
    torch.manual_seed(args.seed)
    device = get_device()
    encoder = Encoder(latent_dim=args.latent_dim)
    if load:
        encoder.load_state_dict(torch.load(args.encoder, map_location='cpu'))
    encoder = encoder.to(device)
    classifier = Classifier(args.latent_dim).to(device)

    trainset, testset = get_data(args.data)
    # same 1000 images for both runs (seeded), sampled without replacement
    selected_idxs = torch.randperm(len(trainset))[:args.num_labels]
    trainset = torch.utils.data.Subset(trainset, selected_idxs.tolist())
    trainloader = get_dataloader(trainset, 64, num_workers=args.workers)
    testloader = get_dataloader(testset, 256, is_testset=True,
                                num_workers=args.workers)

    # epoch 0: classifier only, encoder untouched
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    train(encoder, classifier, trainloader, optimizer, device)
    accs = [accuracy(encoder, classifier, testloader, device)]
    print(f'Epoch 0 - Accuracy: {accs[-1]:.4f}')

    # epochs 1..N: fine-tune everything with one optimizer
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()), lr=3e-4)
    for epoch in range(1, args.epochs + 1):
        train(encoder, classifier, trainloader, optimizer, device)
        accs.append(accuracy(encoder, classifier, testloader, device))
        print(f'Epoch {epoch} - Accuracy: {accs[-1]:.4f}')
    return accs


def main(args):
    if not os.path.exists(args.encoder):
        raise SystemExit(f'{args.encoder} not found: run version3.py first.')
    plt.plot(run(args, load=False))
    plt.plot(run(args, load=True))
    plt.legend(['Random init', 'MoCo pretrained'])
    plt.xlabel('Epoch')
    plt.ylabel('Test accuracy')
    plt.title(f'MoCo + fine-tuning with {args.num_labels} labels')
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, 'finetune.png')
    plt.savefig(path)
    print(f'Saved {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data', default='./data')
    parser.add_argument('--encoder', default='./checkpoints/moco_encoder.pth')
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--num-labels', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--out', default='./outputs')
    parser.add_argument('--seed', type=int, default=2023)
    main(parser.parse_args())
