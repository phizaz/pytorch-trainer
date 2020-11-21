from mlkit.nlp.lm import PennTreebank
from mlkit.nlp.lm.torch_lstm import RNNModel
from mlkit.start import *
from mlkit.trainer.apex_trainer import *
from mlkit.trainer.start import *

device = 'cuda:0'
dataset = PennTreebank(32, 35, device)

n_token = dataset.n_token

n_in = 400
n_hid = 400
n_layer = 1
drop_layer = 0.25
drop_emb = 0.1
lr = 1e-3

def net_fn():
    return RNNModel(
        'LSTM',
        n_token,
        n_in,
        n_hid,
        n_layer,
        c_drop_layer=drop_layer,
        c_drop_emb=drop_emb,
        tie_weights=True,
    )

def opt_fn(net):
    return optim.Adam(net.parameters(), lr=lr)

def loss_fn(pred, y):
    pred = pred.view(-1, n_token)
    y = y.view(-1)
    return F.cross_entropy(pred, y)

use_apex = True

cls = LMTrainer
if use_apex:
    cls = apex_trainer_mask(cls, opt_level='O1')

trainer = cls(net_fn, opt_fn, device, loss_fn)
stats = trainer.train(
    dataset.train_loader,
    10_000,
    callbacks=trainer.make_default_callbacks() + [
        GradientClipCb(0.25),
    ]
)
