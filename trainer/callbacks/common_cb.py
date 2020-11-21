from torch import optim

from ..params_grads import *
from .base_cb import *

class GradientClipCb(Callback):
    """Supports Apex's AMP"""
    def __init__(self, clip_norm, **kwargs):
        super().__init__(**kwargs)
        self.clip_norm = clip_norm

    def on_backward_end(self, trainer, **kwargs):
        nn.utils.clip_grad_norm_(iter_opt_params(trainer.opt), max_norm=self.clip_norm)

class LRSchedulerCb(Callback):
    """
    Args:
        lr_fn: learning rate function (p, i, i_ep, loss) -> (float, None); None to ignore.
    """
    def __init__(self, lr_fn, **kwargs):
        super().__init__(**kwargs)
        self.lr_fn = lr_fn

    def _get_stats(self, callbacks, key):
        for cb in callbacks:
            if isinstance(cb, StatsCallback):
                if key in cb.stats:
                    v = cb.stats[key]
                    return v
        raise NotImplementedError(f'{key} not found')

    def on_step_begin(self, trainer, n_max_itr, i_itr, n_ep_itr, callbacks, **kwargs):
        loss = self._get_stats(callbacks, 'loss')
        # f_ep should start from 0.00 and is float
        f_ep = i_itr / n_ep_itr
        n_max_ep = n_max_itr / n_ep_itr
        scale = self.lr_fn(
            p=i_itr / n_max_itr,
            i=i_itr,
            n_max_itr=n_max_itr,
            f_ep=f_ep,
            n_max_ep=n_max_ep,
            loss=loss,
        )
        # only care the float scales
        if isinstance(scale, (int, float)):
            for g in trainer.opt.param_groups:
                assert 'lr' in g, "the optimizer doesn't seed to have the lr option"
                if 'base_lr' not in g:
                    g['base_lr'] = g['lr']
                g['lr'] = g['base_lr'] * scale

class LRReducePlateauCb(Callback):
    """
    Args:
        key: the key to watch
        n_cycle: how frequent to check (need to match the validator), default: n_ep_itr
        **kwargs: see ReduceLROnPlateau on Pytorch
    """
    def __init__(self, key, n_cycle=None, mode='max', patience=10, factor=0.2, **kwargs):
        super().__init__()
        self.scheduler = None
        self.key = key
        self.n_cycle = n_cycle
        self.mode = mode
        self.patience = patience
        self.factor = factor
        self.kwargs = kwargs

    def get_state(self):
        assert self.scheduler is not None
        return {
            'self': super().get_state(),
            'scheduler': self.scheduler.state_dict(),
        }

    def load_state(self, state):
        assert self.scheduler is not None
        super().load_state(state['self'])
        self.scheduler.load_state_dict(state['scheduler'])

    # should load before resume
    @set_order(0)
    def on_train_begin(self, trainer, n_ep_itr, **kwargs):
        if self.n_cycle is None:
            # set default to 1 epoch
            self.n_cycle = n_ep_itr

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            trainer.opt,
            mode=self.mode,
            patience=self.patience,
            factor=self.factor,
            **self.kwargs,
        )

    def on_batch_end(self, callbacks, i_itr, **kwargs):
        if i_itr % self.n_cycle == 0:
            # getting the key value from the stats
            v = None
            for cb in callbacks:
                if isinstance(cb, StatsCallback):
                    if self.key in cb.stats:
                        v = cb.stats[self.key]
                        break
            assert v is not None, "needs to set the cycle to match the validator callback"
            self.scheduler.step(v)

class WeightPolyakCb(Callback):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.w = None
        self.rate = rate

    def get_params(self, net):
        return nn.utils.parameters_to_vector(net.parameters())

    def on_step_begin(self, trainer, **kwargs):
        # get params
        self.w = self.get_params(trainer.net)

    @torch.no_grad()
    def on_step_end(self, trainer, **kwargs):
        # update w
        new_w = self.get_params(trainer.net)
        new_w = self.rate * self.w + (1 - self.rate) * new_w
        nn.utils.vector_to_parameters(new_w, trainer.net.parameters())
        self.w = None

class TerminateLRCb(Callback):
    def __init__(self, lr_thresh, begin=0):
        super().__init__()
        self.lr_thresh = lr_thresh
        self.begin = begin

    def on_batch_end(self, trainer, i_itr, **kwargs):
        if i_itr >= self.begin:
            lr = trainer.opt.param_groups[0]['lr']
            if lr <= self.lr_thresh:
                print(f'terminated because lr {lr} <= {self.lr_thresh}')
                raise KeyboardInterrupt()

class StopAnyTimeCb(Callback):
    """supress the keyboard interrupt allowing to stop the training anytime
    while getting the return results"""
    def on_abrupt_end(self, e, **kwargs):
        if isinstance(e, KeyboardInterrupt):
            # suppress the raise
            return True
        else:
            # cannot suppress errors
            return False

class AutoInterrupt(Callback):
    """raises a KeyboardInterrupt at n_itr, useful for playing around."""
    def __init__(self, n_itr, order=None):
        super().__init__(order=order)
        self.n_itr = n_itr

    def on_batch_begin(self, i_itr, **kwargs):
        # this will allow for the validate to end from the last itr
        if i_itr >= self.n_itr:
            raise KeyboardInterrupt()