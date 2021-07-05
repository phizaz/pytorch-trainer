from collections import defaultdict
from dataclasses import dataclass

from torch import nn, optim
from torch.utils.data import DataLoader

from trainer.callbacks.exceptions import *

from .callbacks.base_cb import Callback, callback_call
from .types import *


@dataclass
class StageVars:
    """
    Aaccessible data for each stage of the training loop
    """
    trainer: 'LooperInterface'
    looper: 'Looper'
    loader: DataLoader
    n_max_itr: int
    n_ep_itr: int
    callbacks: List['Callback']
    i_ep: int
    f_ep: float
    p_ep: float
    buffer: Dict
    i_itr: int
    i_sample: int
    data: Dict = None
    forward: Dict = None
    e: Exception = None


class LooperInterface:
    """a base for looper should have the following interface"""
    def __init__(self):
        # book keeping
        self.state = {'i_itr': 0, 'i_sample': 0}
        # buffer collects outputs from the model
        # this is useful for calculating dataset-wise metrics, like BLEU or AUROC
        # callbacks populate data in buffer
        self.buffer = defaultdict(list)
        self.net: nn.Module = None
        self.opt: optim.Optimizer = None

    def on_train_begin(self, vars: StageVars):
        pass

    def on_ep_begin(self, vars: StageVars):
        pass

    def forward_pass(self, vars: StageVars):
        pass

    def backward_pass(self, vars: StageVars):
        pass

    def optimize(self, vars: StageVars):
        pass

    def on_abrupt_end(self, vars: StageVars):
        pass


class Looper:
    """
    looper lopps over a loader with predefined number of iterations
    Goal: removing the duplicated parts between trainer and predictor

    Args:
        base: a class with "on_ep_begin", "forward_pass", "backward_pass", "optimize" methods
        net: the model
        mode: 'train' or 'eval'
    """
    def __init__(
            self,
            base: LooperInterface,
            callbacks: List[Callback],
    ):
        self.base = base
        self.callbacks = callbacks

    @property
    def state(self):
        # state is kept in the base
        return self.base.state

    @property
    def buffer(self):
        # buffer is kept in the base
        return self.base.buffer

    def stage_vars(self, data: Dict = None, forward: Dict = None, e: Exception = None):
        """these will be supplied to callbacks and method calls,
        these are variables that are expected to be used by any callback"""
        n_ep_itr = len(self.loader)
        vars = StageVars(
            trainer=self.base,
            looper=self,
            loader=self.loader,
            n_max_itr=self.n_max_itr,
            n_ep_itr=n_ep_itr,
            callbacks=self.callbacks,
            i_ep=int(self.state['i_itr'] / n_ep_itr) + 1,
            f_ep=self.state['i_itr'] / n_ep_itr,
            p_ep=(self.state['i_itr'] % n_ep_itr) / n_ep_itr * 100,
            buffer=self.buffer,
            i_itr=self.state['i_itr'],
            i_sample=self.state['i_sample'],
            data=data,
            forward=forward,
            e=e,
        )
        return vars

    def one_batch(self, data: Dict):
        """runs for one batch"""
        # start of iteration
        self.state['i_itr'] += 1
        self('on_batch_begin', data=data)
        # forward pass
        self('on_forward_begin', data=data)
        forward = self.base.forward_pass(data=data, vars=self.stage_vars(data=data))
        assert 'n' in forward, "forward must contain 'n'"
        self.state['i_sample'] += forward['n']
        self('on_forward_end', data=data, forward=forward)
        # backward pass
        self('on_backward_begin', data=data, forward=forward)
        self.base.backward_pass(vars=self.stage_vars(data=data, forward=forward))
        self('on_backward_end', data=data, forward=forward)
        # step the optimizer
        self('on_step_begin', data=data, forward=forward)
        self.base.optimize(vars=self.stage_vars(data=data, forward=forward))
        self('on_step_end', data=data, forward=forward)
        self('on_batch_end', data=data, forward=forward)

        # iteration exceeds
        if self.state['i_itr'] >= self.n_max_itr:
            raise StopIteration()

    def one_epoch(self, loader: DataLoader):
        """runs for one epoch"""
        self.base.on_ep_begin(vars=self.stage_vars())
        self('on_ep_begin')
        try:
            for data in loader:
                self.one_batch(data)
        except StopIteration:
            # normal stop
            self('on_ep_end')
        except GracefulException:
            # graceful stop, no real problem
            self('on_ep_end')
            raise
        except KeyboardInterrupt:
            self('on_ep_end')
            raise
        except Exception:
            raise

    def loop(self, loader: DataLoader, n_max_itr: int):
        """main method to loop over the dataloader"""
        self.loader = loader
        self.n_max_itr = n_max_itr

        try:
            self.base.on_train_begin(vars=self.stage_vars())
            self('on_train_begin')
            while self.state['i_itr'] < self.n_max_itr:
                self.one_epoch(self.loader)
            self('on_train_end')
        except GracefulException:
            print('graceful exception')
            self('on_train_end')
        except KeyboardInterrupt as e:
            # run the train_end before exiting (saving etc.)
            print('keyboard interrupt - wait for the finalizations')
            # this allows the callback to suppress the keyboard interrupt
            self.base.on_abrupt_end(vars=self.stage_vars(e=e))
            if not self('on_abrupt_end', e=e):
                raise e
        except Exception as e:
            # in other cases, we should not call the train_end to slow things down
            print('unexpected exception:', e)
            self.base.on_abrupt_end(vars=self.stage_vars(e=e))
            self('on_abrupt_end', e=e)
            raise e

    def __call__(self, event, **kwargs):
        """call event callbacks"""
        return callback_call(callbacks=self.callbacks,
                             method=event,
                             vars=self.stage_vars(**kwargs))

