from torch.utils.data import DataLoader

from ..predictor_validate import ValidatePredictor
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
        n_ep_cycle: float = None,
        n_sample_cycle: int = None,
        on_end=False,
        predictor_cls=ValidatePredictor,
    ):
        # n_log_cycle = 1, when it say writes it should write
        super().__init__(n_itr_cycle=n_itr_cycle,
                         n_ep_cycle=n_ep_cycle,
                         n_sample_cycle=n_sample_cycle)
        self.loader = loader
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
            ProgressCb(self.name),
            ReportLoaderStatsCb(),
        ]

    def on_train_begin(self, vars: StageVars):
        super().on_train_begin(vars)
        if self.n_itr_cycle is None:
            if self.n_ep_cycle is not None:
                self.n_itr_cycle = int(self.n_ep_cycle * vars.n_ep_itr)
            else:
                # default to 1 ep
                self.n_itr_cycle = vars.n_ep_itr

    # should run a bit early
    # so that others that might be able to use 'val_loss'
    @set_order(90)
    def on_batch_end(self, vars: StageVars):
        if ((self.on_end and vars.i_itr == vars.n_max_itr)
                or vars.i_itr % self.n_itr_cycle == 0):

            # make prediction and collect the stats
            # predictor returns the stats
            predictor = self.predictor_cls(vars.trainer,
                                           callbacks=self.callbacks)
            res = predictor.predict(self.loader)
            if isinstance(res, tuple):
                bar, info = res
                assert isinstance(bar, dict)
                assert isinstance(info, dict)
            elif isinstance(res, dict):
                bar = res
                info = {}
            else:
                raise NotImplementedError()

            # prepend the keys with its name
            def prepend(d):
                out = {}
                for k, v in d.items():
                    out[f'{self.name}_{k}'] = v
                return out

            bar = prepend(bar)
            info = prepend(info)

            bar.update({'i_itr': vars.i_itr})
            info.update({'i_itr': vars.i_itr})
            self.add_to_bar_and_hist(bar)
            self.add_to_hist(info)
            self._flush()
        super().on_batch_end(vars)
