from torch import nn, optim
from torchvision.datasets import MNIST
from torchvision.transforms import *
from trainer.start import *


def net_fn():
    return nn.Sequential(
        nn.Linear(28 * 28, 300),
        nn.ReLU(),
        nn.Linear(300, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )


def opt_fn(net):
    return optim.Adam(net.parameters(), lr=1e-3)


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    dataset_path = f'{dirname}/dataset/mnist'

    train_dataset = MNIST(dataset_path,
                          train=True,
                          transform=ToTensor(),
                          download=True)
    test_dataset = MNIST(dataset_path,
                         train=False,
                         transform=ToTensor(),
                         download=True)

    train_loader = FlattenLoader(
        DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4))
    test_loader = FlattenLoader(
        DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4))

    device = 'cuda:0'
    train_loader = ConvertLoader(train_loader, device)
    test_loader = ConvertLoader(test_loader, device)

    make_trainer = apex_trainer_mask(MultiClassTrainer, opt_level='O1')

    val_cb = ValidateCb(test_loader, callbacks=AvgCb(['loss', 'acc']))
    cb = make_trainer.make_default_callbacks() + [val_cb]
    trainer = make_trainer(net_fn, opt_fn, device, cb)
    df = trainer.train(train_loader, n_max_ep=2)

    print(val_cb.last_hist)
    assert val_cb.last_hist['val_acc'] > 0.9