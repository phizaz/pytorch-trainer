from sklearn.metrics import roc_auc_score

from ..average import *
from ..types import *
from .report_cb import *

class CollectCb(StatsCallback):
    """collect the prediction of a whole episode"""
    def __init__(self, keys):
        super().__init__()
        self.collect_keys = keys
        # should this callback be the one to collect?
        self.has_authority = {k: False for k in self.collect_keys}

    def on_ep_begin(self, buffer, **kwargs):
        for k in self.collect_keys:
            if k not in buffer:
                self.has_authority[k] = True
            if self.has_authority[k]:
                buffer[k] = []

    def _combine(self, v):
        try:
            return torch.cat(v)
        except Exception:
            # batch data is a list
            # flatten 2d list
            out = []
            for x in v:
                for y in x:
                    out.append(y)
            return out

    def on_forward_end(self, forward, buffer, i_itr, n_ep_itr, **kwargs):
        for k in self.collect_keys:
            if self.has_authority[k]:
                buffer[k].append(cpu(detach(forward[k])))
                # flattens on the end of epoch
                if i_itr % n_ep_itr == 0:
                    buffer[k] = self._combine(buffer[k])

class MovingAvgCb(BoardCallback):
    """
    a moving average, doesn't reset between epochs
    """
    def __init__(self, keys=['loss'], prefix='', n=100, mode='bar', **kwargs):
        super().__init__(**kwargs)
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        self.prefix = prefix
        self.mode = mode
        # average is the state
        self._state['avg'] = defaultdict(partial(SMA, size=n))

    def on_backward_begin(self, forward, i_itr, **kwargs):
        # update the stats
        for k in self.keys:
            val = forward.get(k, None)
            if val is None: continue
            if isinstance(val, Tensor) and len(val.shape) == 1:
                val = val.mean()
            self.avg[k].update(val, w=forward['n'])

        info = {f'{self.prefix}{k}': self.avg[k].val() for k in self.keys}
        info['i_itr'] = i_itr
        if self.mode == 'bar':
            self.add_to_bar_and_hist(info)
        elif self.mode == 'buffer':
            self.add_to_hist(info)
        else:
            raise NotImplementedError()

class AvgCb(StatsCallback):
    """
    AvgCb uses average, and will reset every epoch, used in validation
    """
    def __init__(self, keys):
        super().__init__()
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        self.avg = None

    def on_ep_begin(self, **kwargs):
        self.avg = defaultdict(Average)

    def on_backward_begin(self, i_itr, forward, **kwargs):
        for k in self.keys:
            val = forward[k].item()
            n = forward['n']
            self.avg[k].update(val, w=n)
        info = {k: self.avg[k].val() for k in self.keys}
        info['i_itr'] = i_itr
        self.add_to_bar_and_hist(info)

class AUROCCb(CollectCb):
    """add area-under-roc-curve during validation
    Args:
        sigmoid: automatically apply sigmoid on the prediction
        auroc_fn: (pred, y) -> {'auroc', 'support'}
        cls_names: a mapping i => string
    """
    def __init__(self, cls=None, apply_sigmoid=True, cls_names=None):
        # collect 'pred' and 'y'
        super().__init__(keys=['pred', 'y'])
        self.sigmoid = apply_sigmoid
        self.cls = cls
        self.cls_names = cls_names

    def on_ep_end(self, buffer, i_itr, **kwargs):
        pred = buffer['pred']
        y = buffer['y']
        assert isinstance(pred, Tensor), 'you forgot to tensorify pred'
        assert isinstance(y, Tensor), 'you forgot to tensorify y'

        if self.sigmoid:
            pred = torch.sigmoid(pred)
        if self.cls is not None:
            pred = pred[:, self.cls]
            y = y[:, self.cls]
            idx = self.cls
        else:
            idx = list(range(y.shape[1]))

        # aurocs
        aurocs = dict()
        support = dict()
        for i in idx:
            aurocs[i] = roc_auc_score(y[:, i], pred[:, i])
            support[i] = y[:, i].sum()
        total = sum(support[i] for i in idx)
        micro = sum(aurocs[i] * support[i] / total for i in idx)
        macro = np.array([aurocs[i] for i in idx]).mean()

        bar = {
            'i_itr': i_itr,
            'auroc_micro': micro,
            'auroc_macro': macro,
        }
        self.add_to_bar_and_hist(bar)

        if self.cls_names is None:
            info = {f'auroc_{i}': aurocs[i] for i in idx}
        else:
            info = {f'auroc_{self.cls_names[i]}': aurocs[i] for i in idx}
        info['i_itr'] = i_itr
        self.add_to_hist(info)
        self._flush()
