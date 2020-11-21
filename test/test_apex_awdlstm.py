from mlkit.nlp.lm import PennTreebank
from mlkit.nlp.lm.awd_lstm.awd_trainer import AWDLMTrainer
from mlkit.nlp.lm.awd_lstm.model import AWD_LSTM
from mlkit.start import *
from mlkit.trainer.apex_trainer import *
from mlkit.trainer.start import *

device = 'cuda:1'
torch.cuda.set_device(device)
dataset = PennTreebank(32, 35, device)

n_token = dataset.n_token

n_hid = 1152
n_inp = 400  # 448
n_layer = 3
drop_in_out = 0.0
drop_layer = 0.0
drop_emb = 0.0
drop_weight = 0.0
alpha = 0
beta = 0
lr = 1e-3

def net_fn():
    return AWD_LSTM(
        'LSTM',
        n_token,
        n_inp,
        n_hid,
        n_layer,
        c_drop_out=drop_in_out,
        c_drop_layer=drop_layer,
        c_drop_in=drop_in_out,
        c_drop_emb=drop_emb,
        c_drop_weight=drop_weight,
        c_alpha=alpha,
        c_beta=beta,
        tie_weights=True,
        use_split_ce=False,
    )

def opt_fn(net):
    return optim.Adam(net.parameters(), lr=lr)

def loss_fn(pred, y):
    pred = pred.view(-1, n_token)
    y = y.view(-1)
    return F.cross_entropy(pred, y)

use_apex = True

cls = AWDLMTrainer
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
