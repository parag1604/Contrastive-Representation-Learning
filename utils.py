"""Shared helpers for the contrastive representation learning tutorial.

Everything that does not change between versions lives here:
- picking a device (cuda -> mps -> cpu),
- the MNIST datasets and data augmentation,
- the convolutional Encoder and the linear Classifier,
- evaluation: the linear probe and the t-SNE plot of the learnt embeddings.

Terminology follows the "Contrastive losses come from energy" section of
https://parag1604.github.io/notes/Machine-Learning/Energy-Based-Models :
an anchor x is scored against candidates y with an energy E(x, y); the
matching candidate is the *positive*, the others are *negatives*, and for
embeddings f(x), f(y) we use E(x, y) = -f(x)^T f(y) / tau, with tau the
*temperature*.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ---------------------------------------------------------------- data
# The augmentation defines what the representation should be *invariant* to:
# small rotations and shifts do not change the identity of a digit.
augment = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(20),
    transforms.RandomCrop((28, 28)),
    transforms.ToTensor(),
])


class TwoViews:
    """Apply the same random transform twice: two *views* of one image.

    Used from version2 onwards, where the positive for an anchor is simply
    another augmentation of the same image (no labels needed).
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)


def get_data(root: str = './data', two_views: bool = False):
    """MNIST is downloaded by torchvision into `root` on the first run."""
    trainset = datasets.MNIST(
        root=root, train=True, download=True,
        transform=TwoViews(augment) if two_views else augment)
    testset = datasets.MNIST(
        root=root, train=False, download=True,
        transform=transforms.ToTensor())
    return trainset, testset


def get_dataloader(dataset: Dataset, batch_size: int, is_testset: bool = False,
                   num_workers: int = 2) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=not is_testset,
                      num_workers=num_workers, drop_last=not is_testset)


def get_label_index(dataset) -> dict:
    """labels[k] = indices of all training images of class k."""
    targets = torch.as_tensor(dataset.targets)
    return {k: torch.where(targets == k)[0].numpy() for k in range(10)}


# ---------------------------------------------------------------- models
class Encoder(nn.Module):
    """Small CNN mapping a 1x28x28 image to a `latent_dim` embedding."""
    def __init__(self, latent_dim: int = 128):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.fc1 = nn.Linear(128 * 3 * 3, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))  # 28 -> 28
        x = F.max_pool2d(x, 2, 2)  # 28 -> 14
        x = F.relu(self.conv2(x))  # 14 -> 14
        x = F.max_pool2d(x, 2, 2)  # 14 -> 7
        x = F.relu(self.conv3(x))  # 7 -> 7
        x = F.max_pool2d(x, 2, 2)  # 7 -> 3
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class Classifier(nn.Module):
    """A single linear layer on top of the embedding (returns log-probs)."""
    def __init__(self, latent_dim: int = 128, out_dim: int = 10) -> None:
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(latent_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.fc1(x), dim=1)


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def accuracy(encoder, classifier, testloader, device) -> float:
    encoder.eval()
    classifier.eval()
    correct = 0
    for x, y in testloader:
        y_pred = classifier(encoder(x.to(device)))
        correct += (y_pred.argmax(dim=1).cpu() == y).sum().item()
    return correct / len(testloader.dataset)


def linear_probe(encoder, trainloader, testloader, device, epochs: int = 3,
                 max_batches: int = None) -> float:
    """Freeze the encoder and train only a linear classifier on top.

    This is the standard way to judge a representation: if the classes are
    linearly separable in embedding space, the encoder has learnt something
    useful *without* ever being told the labels (from version2 onwards).
    """
    classifier = Classifier(encoder.latent_dim).to(device)
    clf_optim = optim.Adam(classifier.parameters(), lr=1e-3)
    encoder.eval()
    acc = 0.
    for epoch in range(epochs):
        classifier.train()
        for batch_idx, (x, y) in enumerate(tqdm(trainloader, desc='probe')):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                z = encoder(x)  # frozen: no gradient into the encoder
            loss = F.nll_loss(classifier(z), y)
            clf_optim.zero_grad()
            loss.backward()
            clf_optim.step()
        acc = accuracy(encoder, classifier, testloader, device)
        print(f'Linear probe epoch {epoch + 1}: test accuracy {acc:.4f}')
    return acc


@torch.no_grad()
def embed(encoder, testloader, device, max_points: int = None):
    encoder.eval()
    zs, ys = [], []
    for x, y in testloader:
        zs.append(encoder(x.to(device)).cpu().numpy())
        ys.append(y.numpy())
        if max_points is not None and sum(map(len, ys)) >= max_points:
            break
    zs, ys = np.concatenate(zs), np.concatenate(ys)
    return zs[:max_points], ys[:max_points]


def plot_tsne(x: np.ndarray, y: np.ndarray, path: str, title: str = '') -> None:
    """2-D t-SNE of the embeddings, coloured by the (unseen) digit label."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tsne = TSNE(n_components=2, perplexity=min(30, len(x) - 1), init='pca')
    x_emb = tsne.fit_transform(x)
    for i in range(10):
        plt.scatter(x_emb[y == i, 0], x_emb[y == i, 1], s=1, label=str(i))
    plt.legend(markerscale=5)
    plt.title(title)
    plt.savefig(path)
    plt.close()
    print(f'Saved t-SNE plot to {path}')


# ---------------------------------------------------------------- CLI glue
def add_common_args(parser):
    parser.add_argument('--data', default='./data', help='MNIST download folder')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--probe-epochs', type=int, default=3,
                        help='epochs of linear-probe training')
    parser.add_argument('--probe-batches', type=int, default=None,
                        help='cap batches per probe epoch (for quick runs)')
    parser.add_argument('--tsne-points', type=int, default=5000,
                        help='test images shown in the t-SNE plot')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--out', default='./outputs', help='plots go here')
    parser.add_argument('--seed', type=int, default=2023)
    return parser


def evaluate(encoder, args, name: str, device) -> float:
    """t-SNE of test embeddings + linear-probe accuracy (labels used only here)."""
    _, testset = get_data(args.data)
    probe_trainset = datasets.MNIST(args.data, train=True, download=True,
                                    transform=augment)
    trainloader = get_dataloader(probe_trainset, args.batch_size,
                                 num_workers=args.workers)
    testloader = get_dataloader(testset, 2 * args.batch_size, is_testset=True,
                                num_workers=args.workers)
    x, y = embed(encoder, testloader, device, max_points=args.tsne_points)
    plot_tsne(x, y, os.path.join(args.out, f'tsne_{name}.png'), title=name)
    return linear_probe(encoder, trainloader, testloader, device,
                        epochs=args.probe_epochs, max_batches=args.probe_batches)
