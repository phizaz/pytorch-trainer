from torch.utils.data import DataLoader

from ..predictor_base import BasePredictor
from .base_cb import *
from .report_cb import *


class ValidateCb(BoardCallback):
    """validate every n iteration,
    it will not report anything by default

    To report loss and accuracy, use:
        callbacks=[AvgCb(['loss', 'acc])] 
    
    Args:
        n_val_cycle: default: n_ep_itr
        on_end: extra validation before ending (even it doesn't divide)
    """
    def __init__(
            self,
            loader: DataLoader,
            callbacks=None,
            name: str = 'val',
            n_itr_cycle: int = None,
            n_ep_cycle: int = None,
            on_end=False,
            predictor_cls=BasePredictor,
    ):
        # n_log_cycle = 1, when it say writes it should write
        super().__init__()
        self.loader = loader
        self.n_itr_cycle = n_itr_cycle
        self.n_ep_cycle = n_ep_cycle
        self.name = name
        if callbacks is None:
            callbacks = []
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        self.callbacks = callbacks + self.make_default_callbacks()
        self.on_end = on_end
        self.predictor_cls = predictor_cls

    def make_default_callbacks(self):
        return [
            ProgressCb(self.name, destroy=False),
            ReportLoaderStatsCb(),
        ]

    def on_train_begin(self, n_ep_itr, **kwargs):
        super().on_train_begin(n_ep_itr=n_ep_itr, **kwargs)
        if self.n_itr_cycle is None:
            if self.n_ep_cycle is not None:
                self.n_itr_cycle = self.n_ep_cycle * n_ep_itr
            else:
                # default to 1 ep
                self.n_itr_cycle = n_ep_itr

    # should run a bit early
    # so that others that might be able to use 'val_loss'
    @set_order(90)
    def on_batch_end(self, trainer, i_itr: int, n_max_itr: int, **kwargs):
        if ((self.on_end and i_itr == n_max_itr)
                or i_itr % self.n_itr_cycle == 0):

            # make prediction
            predictor = self.predictor_cls(trainer,
                                           callbacks=self.callbacks,
                                           collect_keys=[])
            predictor.predict(self.loader)

            # collect all data from callbacks
            out = StatsCallback.collect_latest(self.callbacks)

            # the keys in the callbacks should be visible on the progress bar
            # everything else is kept in the buffer (not visible)
            bar_keys = set()
            for cb in self.callbacks:
                if isinstance(cb, StatsCallback):
                    bar_keys |= set(cb.stats.keys())
            bar_keys -= set(['i_itr'])

            bar = {'i_itr': i_itr}
            info = {'i_itr': i_itr}
            for k, v in out.items():
                if k in bar_keys:
                    bar[f'{self.name}_{k}'] = v
                else:
                    info[f'{self.name}_{k}'] = v
            self.add_to_bar_and_hist(bar)
            self.add_to_hist(info)
            self._flush()
