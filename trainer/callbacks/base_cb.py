import os
from collections import defaultdict
from functools import partial
from shutil import rmtree

import pandas as pd
import torch

from ..numpy_writer import *
from ..params_grads import *
from ..save import *
from ..stateful import *
from ..types import *


def set_order(order: float):
    """decorator to set callback's method order
    usage: 
        @set_order(100)
        def method(self):
            pass
    """
    def inner(meth):
        def fn(*args, **kwargs):
            return meth(*args, **kwargs)

        fn._order = order
        return fn

    return inner


class Callback(Stateful):
    """
    when not to use callbacks:
    - if it is required for correct forward pass of a model, that should be in trainer
    """
    _order = 100

    def save(self, path: str):
        """save state to file"""
        if self.is_state_empty():
            # don't need to save empty state
            return
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        safe_torch_save(self.get_state(), path)

    def load(self, path: str, map_location=None):
        """load state from file"""
        if self.is_state_empty():
            # this cb doesn't need a state
            # caution: cb that needs a state must have the "footprint"
            # of the states, so that it would not be empty at first!
            # unless it will not be loaded!
            return
        self.load_state(torch.load(path, map_location=map_location))

    def on_train_begin(self, vars: 'StageVars'):
        pass

    def on_ep_begin(self, vars: 'StageVars'):
        pass

    def on_batch_begin(self, vars: 'StageVars'):
        pass

    def on_forward_begin(self, vars: 'StageVars'):
        pass

    def on_forward_end(self, vars: 'StageVars'):
        pass

    def on_backward_begin(self, vars: 'StageVars'):
        pass

    def on_backward_end(self, vars: 'StageVars'):
        pass

    def on_step_begin(self, vars: 'StageVars'):
        pass

    def on_step_end(self, vars: 'StageVars'):
        pass

    def on_batch_end(self, vars: 'StageVars'):
        pass

    def on_ep_end(self, vars: 'StageVars'):
        pass

    def on_train_end(self, vars: 'StageVars'):
        pass

    def on_abrupt_end(self, vars: 'StageVars'):
        pass

    def __str__(self):
        return self.__repr__()


class StatsCallback(Callback):
    """
    base class for callbacks that keep stats as "history" 
    """
    def __init__(
        self,
        n_itr_cycle: int = None,
        n_ep_cycle: float = None,
        n_sample_cycle: int = None,
    ):
        super().__init__()
        self.n_itr_cycle = n_itr_cycle
        self.n_ep_cycle = n_ep_cycle
        self.n_sample_cycle = n_sample_cycle

        self._state['hist'] = defaultdict(list)
        self._state['prev_i_sample'] = 0
        # to be put to the progress bar (only)
        self.stats = {}
        # to be put to the history
        # we have buffer so that, we can update many times per iteration
        # not need to collect everything first
        self.buffer = {}

    @classmethod
    def collect_latest(cls, callbacks):
        """collect latest entries from all callbacks (that is StatsCallback), excluding i_itr"""
        out = {}
        for cb in callbacks:
            if isinstance(cb, StatsCallback):
                for k, v in cb.last_hist.items():
                    if k != 'i_itr':
                        out[k] = v
        return out

    @classmethod
    def combine_callbacks(cls, callbacks):
        """merge dataframes from callbacks"""
        out = None
        for cb in callbacks:
            if isinstance(cb, StatsCallback):
                df = cb.df
                if 'i_itr' in df:
                    if out is None:
                        out = cb.df
                    else:
                        out = pd.merge(out, cb.df, on='i_itr', how='outer')
        return out

    @property
    def hist(self):
        return self._state['hist']

    @property
    def last_hist(self) -> Dict:
        return {k: v[-1] for k, v in self.hist.items()}

    @property
    def df(self):
        return pd.DataFrame(self.hist)

    def on_train_begin(self, vars: 'StageVars'):
        """automatically gets the ep cycle if n_itr_cycle is not given"""
        if self.n_itr_cycle is None:
            if self.n_ep_cycle is not None:
                self.n_itr_cycle = int(self.n_ep_cycle * vars.n_ep_itr)
            elif self.n_sample_cycle is not None:
                # do nothing .. we will use the vars.i_sample
                pass
            else:
                # default to 1 itr
                self.n_itr_cycle = 1

    def on_batch_begin(self, vars: 'StageVars'):
        # clear the buffer
        self.buffer = {}

    def on_batch_end(self, vars: 'StageVars'):
        """auto-flush after each iteration.
        don't forget to flush if you overload this method."""
        # update the prev
        if self.n_sample_cycle is not None:
            self.prev_i_sample = vars.i_sample
        self._flush()

    def add_to_bar(self, d, vars: 'StageVars'):
        """update the stats which shall be shown in the progress bar (only)"""
        if 'i_itr' not in d:
            d['i_itr'] = vars.i_itr
        if self.is_log_cycle(vars):
            d = self._eval(d)
            self.stats.update(_strip(d))

    def add_to_bar_and_hist(self, d, vars: 'StageVars'):
        """both update the progress bar and write to the buffer (history), don't forget to flush"""
        if 'i_itr' not in d:
            d['i_itr'] = vars.i_itr
        if self.is_log_cycle(vars):
            d = self._eval(d)
            self.stats.update(_strip(d))
            self.buffer.update(_strip(d))

    def add_to_hist(self, d, vars: 'StageVars'):
        """buffer before putting into the history after flushing"""
        if 'i_itr' not in d:
            d['i_itr'] = vars.i_itr
        if self.is_log_cycle(vars):
            d = self._eval(d)
            self.buffer.update(_strip(d))

    def _flush(self):
        """save the buffer to history"""
        d = self.buffer
        if len(d) > 0:
            assert 'i_itr' in d, f'i_itr is not present in {self}'
            _append_dict(self.hist, d)
            # should not clear the buffer,
            # it might be used by others

    def is_log_cycle(self, vars: 'StageVars', include_first: bool = False):
        """
        it is possible that n_itr_cycle is still None.
        This is the case for n_sample_cycle is set.
        In this case, n_sample_cycle will only be set after the first forward pass.
        """
        if self.n_itr_cycle is not None:
            return vars.i_itr % self.n_itr_cycle == 0 or (include_first
                                                          and vars.i_itr <= 1)
        if self.n_sample_cycle is not None:
            return vars.i_sample // self.n_sample_cycle > self.prev_i_sample // self.n_sample_cycle or (
                include_first and self.prev_i_sample == 0)
        raise NotImplementedError()

    def _eval(self, d):
        for k, v in d.items():
            d[k] = _get_val(v)
        return d


def _strip(x):
    """remvoe tensor-hood from the input structure"""
    if isinstance(x, Tensor):
        x = x.item()
    elif isinstance(x, dict):
        x = {k: _strip(v) for k, v in x.items()}
    return x


class BoardCallback(StatsCallback):
    """writes into a tensorboard"""
    def __init__(
        self,
        n_itr_cycle: int = None,
        n_ep_cycle: float = None,
        n_sample_cycle: int = None,
    ):
        super().__init__(n_itr_cycle=n_itr_cycle,
                         n_ep_cycle=n_ep_cycle,
                         n_sample_cycle=n_sample_cycle)
        self.writer = None

    def on_train_begin(self, vars: 'StageVars'):
        """automatically discovers the tensorboard cb"""
        super().on_train_begin(vars)

        # obtain the summary writer
        for cb in vars.callbacks:
            if isinstance(cb, TensorboardCb):
                self.writer = cb.writer

    def add_to_bar(self, d, vars: 'StageVars'):
        """update the stats which shall be shown in the progress bar (only)"""
        self.add_to_board(d, vars)
        super().add_to_bar(d, vars)

    def add_to_bar_and_hist(self, d, vars: 'StageVars'):
        """both update the progress bar and write to the buffer (history), don't forget to flush"""
        self.add_to_board(d, vars)
        super().add_to_bar_and_hist(d, vars)

    def add_to_hist(self, d, vars: 'StageVars'):
        """buffer before putting into the history after flushing"""
        self.add_to_board(d, vars)
        super().add_to_hist(d, vars)

    def add_to_board(self, d, vars: 'StageVars'):
        """write a dictionary to tensorboard"""
        if self.is_log_cycle(vars):
            d = self._eval(d)
            for k, v in d.items():
                self.add_to_board_scalar(k, v, vars)

    def add_to_board_scalar(self, name, val, vars: 'StageVars'):
        """write a scalar to tensorboard"""
        if self.is_log_cycle(vars):
            if self.writer is not None:
                # tensorboard uses i_sample as the main axis
                self.writer.add_scalar(name, _get_val(val), vars.i_sample)


class NumpyWriterCb(Callback):
    """if this is present, boardcallback will write into the tensorboard"""
    # make sure it initializes before others use it
    _order = 90

    def __init__(self, path, n_max_width=1000, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.n_max_width = n_max_width
        self.np_writer = None

    def on_train_begin(self, vars: 'StageVars'):
        super().on_train_begin(vars)
        self.np_writer = NumpyWriter(self.path, n_max_width=self.n_max_width)

    def on_train_end(self, vars: 'StageVars'):
        super().on_train_end(vars)
        if self.np_writer is not None:
            self.np_writer.flush()
            self.np_writer.close()


class BaseNumpyWriterCb(Callback):
    def __init__(self, n_log_hist_cycle: int):
        super().__init__()
        self.np_writer = None
        self.n_log_hist_cycle = n_log_hist_cycle

    def on_train_begin(self, vars: 'StageVars'):
        """automatically discovers the tensorboard cb"""
        super().on_train_begin(vars)
        for cb in vars.callbacks:
            if isinstance(cb, NumpyWriterCb):
                self.np_writer = cb.np_writer

    def write_hist(self, name, val, i_itr):
        if i_itr % self.n_log_hist_cycle == 0:
            if self.np_writer is not None:
                self.np_writer.write_hist(name, _get_val(val), i_itr)

    def on_batch_end(self, vars: 'StageVars'):
        if self.np_writer is not None:
            self.np_writer.flush()
        super().on_batch_end(vars)


class NumpyWeightHistCb(BaseNumpyWriterCb):
    def __init__(self, n_log_hist_cycle: int):
        super().__init__(n_log_hist_cycle)

    def on_batch_end(self, vars: 'StageVars'):
        self.write_hist('weight',
                        lambda: params_to_vec(vars.trainer.net.parameters()),
                        vars.i_itr)
        super().on_batch_end(vars)


class TensorboardCb(Callback):
    """if this is present, boardcallback will write into the tensorboard
    the path will be extended with a unique random string.
    
    Args:
        resume: if True, use the previous random string; else use a new random string
    """
    def __init__(self, path, resume=True):
        super().__init__()
        self.path = path
        self.resume = resume
        self.writer = None

    # make sure it initializes before others (normal) use it
    # but it should be "after" the autoresume
    @set_order(91)
    def on_train_begin(self, vars: 'StageVars'):
        from torch.utils.tensorboard import SummaryWriter
        super().on_train_begin(vars)
        if not self.resume:
            if os.path.exists(self.path):
                rmtree(self.path)
            else:
                os.makedirs(self.path)
        self.writer = SummaryWriter(self.path, flush_secs=10)

    def on_train_end(self, vars: 'StageVars'):
        if self.writer is not None:
            self.writer.close()
        super().on_train_end(vars)


def get_val_from_statcbs(key, callbacks):
    for cb in callbacks:
        if isinstance(cb, StatsCallback):
            if key in cb.stats:
                v = cb.stats[key]
                return v
    raise ValueError(f'{key} not found')


def _get_val(v):
    """get val from a function or a value"""
    if callable(v):
        return v()
    return v


def _append_dict(dict_of_list, dict):
    """
    append a dict into a dict of lists
    before doing that, all lists should have the same size first!
    append None to the smaller lists.
    """
    def fill_na():
        lengths = [len(v) for v in dict_of_list.values()]
        if len(lengths) == 0:
            max_len = 0
        else:
            max_len = max(lengths)

        # equate the dict sizes with None
        for k in dict_of_list:
            while len(dict_of_list[k]) < max_len:
                dict_of_list[k].append(None)

    fill_na()
    for k, v in dict.items():
        dict_of_list[k].append(v)
    fill_na()


def callback_call(callbacks: List[Callback], method: str, vars: 'StageVars'):
    """call a list of callbacks"""
    if callbacks is None:
        return
    if not isinstance(callbacks, list):
        callbacks = [callbacks]
    # ignore None callbacks
    callbacks = [cb for cb in callbacks if cb is not None]

    # the final return is the "OR" of all return values
    out = None
    for cb in sorted(callbacks, key=partial(_get_cb_order, meth=method)):
        fn = getattr(cb, method, None)
        assert fn is not None, f'the callback {cb} does not have {method}'
        if fn is not None:
            try:
                res = fn(vars=vars)
                assert res is None or isinstance(
                    res, bool
                ), f'returns from the callback {cb} must be either None or a boolean'
            except TypeError as e:
                print(f'type error: {e} ... at {cb}')
                raise e
            except Exception as e:
                print(f'error {e} ... at {cb}')
                raise e

            if res is not None:
                if out is None: out = res
                out |= res
    return out


def _get_cb_order(cb, meth):
    fn = getattr(cb, meth, None)
    if fn is None:
        return cb._order
    # return the method's order (if not use the cb's order)
    order = getattr(fn, '_order', cb._order)
    return order


from ..looper import StageVars

if __name__ == "__main__":
    a = StatsCallback()
    a.add_to_bar_and_hist({'a': 10, 'i_itr': 1})
    print(a.stats)
