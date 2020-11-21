from mlkit.trainer.start import *
from mlkit.vision.data.mnist import MNIST
from mlkit.trainer.apex_trainer import *

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

dirname = os.path.dirname(__file__)

trainer = apex_trainer_mask(
    SimpleTrainer, opt_level='O2'
)(net_fn, opt_fn, 'cuda', F.cross_entropy)

dataset = MNIST('cuda')
trainer.train(
    FlattenLoader(dataset.train_loader(128)),
    10_000,
    callbacks=trainer.make_default_callbacks() + [
        AutoResumeCb(os.path.join(dirname, 'tmp', 'apex'), 500),
        ValidateCb(FlattenLoader(dataset.test_loader(128)), 1000, keys=['loss', 'acc']),
    ]
)
