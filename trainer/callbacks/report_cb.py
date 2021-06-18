import time
from collections import deque

from tqdm.autonotebook import tqdm
from trainer.average import SMA

from ..csv import *
from ..loader_base import BaseLoaderWrapper
from ..params_grads import *
from .base_cb import *

# hierarchy of tqdms
TQDM = []


class ReportItrCb(BoardCallback):
    def on_batch_begin(self, vars: StageVars):
        super().on_batch_begin(vars)
        self.add_to_bar_and_hist({
            'i_itr': vars.i_itr,
            'f_ep': vars.f_ep,
        })
        self.add_to_hist({
            'i_itr': vars.i_itr,
            'i_ep': vars.i_ep,
            '%ep': vars.p_ep
        })


class ProgressCb(Callback):
    """call and collect stats from StatsCallback
    """
    def __init__(self, desc: str = 'train', **kwargs):
        super().__init__(**kwargs)
        self.desc = desc
        self.progress = None

    def update(self, callbacks, i_itr):
        """update the progress bar"""
        stats = {}
        for cb in callbacks:
            if isinstance(cb, StatsCallback):
                stats.update(cb.stats)

        # prevent 1e4 for int
        for k, v in stats.items():
            if isinstance(v, int):
                stats[k] = str(v)

        # don't show i_itr
        if 'i_itr' in stats:
            del stats['i_itr']

        # set postfix must not force the refresh
        self.progress.set_postfix(stats, refresh=False)
        self.progress.update(i_itr - self.progress.n)

    def on_train_begin(self, vars: StageVars):
        # minitirs = 1, means check the update every iteration (disable dynamic miniters)
        # position = len(TQDM) # not friendly
        position = 0  # friendly with logging to file
        self.progress = tqdm(total=vars.n_max_itr,
                             position=position,
                             desc=self.desc,
                             mininterval=0.1,
                             miniters=1)
        TQDM.append(self.progress)

    def close(self):
        self.progress.close()
        TQDM.remove(self.progress)
        self.progress = None

    @set_order(1000)  # wait for stats
    def on_batch_end(self, vars: StageVars):
        self.update(vars.callbacks, vars.i_itr)

    @set_order(1000)  # wait for stats
    def on_train_end(self, vars: StageVars):
        if self.progress is not None:
            self.close()

    def on_abrupt_end(self, vars: StageVars):
        if self.progress is not None:
            self.close()


class ReportLoaderStatsCb(StatsCallback):
    def on_batch_begin(self, vars: StageVars):
        if isinstance(vars.loader, BaseLoaderWrapper):
            # push the value to progress bar
            stats = vars.loader.stats()
            stats['i_itr'] = vars.i_itr
            self.add_to_bar(stats)


class LiveDataframeCb(StatsCallback):
    """pulls data from callbacks and write it into CSV in real-time"""
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.writer = None

    def on_train_begin(self, vars: StageVars):
        if vars.i_itr == vars.n_max_itr:
            # the job is already finished!
            return

        # rewrite all to the file
        df = StatsCallback.combine_callbacks(vars.callbacks)
        self.writer = FastCSVWriter(self.path)
        if df is not None:
            self.writer.write_df(df)

    def write(self, callbacks, i_itr):
        row = {}
        for cb in callbacks:
            if isinstance(cb, StatsCallback):
                # check if the latest entry is the one to be logged
                if len(cb.hist['i_itr']) > 0 and cb.hist['i_itr'][-1] == i_itr:
                    for k, v in cb.hist.items():
                        row.update({k: v[-1]})

        # self.writer.writekvs(row)
        self.writer.write(row)

    # after normal callbacks
    @set_order(110)
    def on_batch_end(self, vars: StageVars):
        self.write(vars.callbacks, vars.i_itr)

    @set_order(110)
    def on_train_end(self, vars: StageVars):
        if self.writer is not None:
            self.write(vars.callbacks, vars.i_itr)
            self.writer.close()


class ReportLRCb(BoardCallback):
    """
    problem: this runs "before" autoresume making it a bit ugly.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_step_end(self, vars: StageVars):
        lrs = []
        for g in vars.trainer.opt.param_groups:
            lrs.append(g['lr'])

        info = {'i_itr': vars.i_itr, 'lr': lrs[0]}
        if len(lrs) > 1:
            info.update({f'lr{i+1}': lr for i, lr in enumerate(lrs[1:])})
        self.add_to_bar_and_hist(info)


class ReportGradnormCb(BoardCallback):
    def __init__(self, name='grad_norm', use_histogram=False, n=100, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.use_histogram = use_histogram
        self._state['avg'] = defaultdict(partial(SMA, size=n))

    def on_backward_end(self, vars: StageVars):
        if self.use_histogram:
            grad_vec = grad_vec_from_params(iter_opt_params(vars.trainer.opt))
            self.add_to_board_histogram(f'{self.name}/hist', grad_vec,
                                        vars.i_itr)
        if self.is_log_cycle(vars.i_itr):
            grad_norm = grads_norm(iter_opt_params(vars.trainer.opt))
            self.avg['grad_norm'].update(grad_norm)
            self.add_to_hist({
                'i_itr': vars.i_itr,
                self.name: self.avg['grad_norm'].val(),
            })


class ReportWeightNormCb(BoardCallback):
    def __init__(self, name='weight_norm', use_histogram=False, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.use_histogram = use_histogram

    def on_step_end(self, vars: StageVars):
        if self.use_histogram:
            self.add_to_board_histogram(
                f'{self.name}/hist', lambda: nn.utils.parameters_to_vector(
                    iter_opt_params(vars.trainer.opt)), vars.i_itr)
        self.add_to_bar_and_hist({
            'i_itr':
            vars.i_itr,
            self.name:
            lambda: many_l2_norm(*list(iter_opt_params(vars.trainer.opt)))
        })


class ReportDeltaNormCb(BoardCallback):
    def __init__(self,
                 name='delta_norm',
                 use_histogram=False,
                 n=100,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.use_histogram = use_histogram
        self.w_before = None
        self._state['avg'] = defaultdict(partial(SMA, size=n))

    def on_backward_end(self, vars: StageVars):
        if self.is_log_cycle(vars.i_itr) or self.is_log_cycle_hist(vars.i_itr):
            # because this command incurs copy of weights in mix precision (could be slow)
            self.w_before = params_to_vec(iter_opt_params(vars.trainer.opt))

    def on_step_end(self, vars: StageVars):
        def delta():
            return self.w_before - params_to_vec(
                iter_opt_params(vars.trainer.opt))

        if self.use_histogram:
            self.add_to_board_histogram(f'{self.name}/hist', delta, vars.i_itr)

        if self.is_log_cycle(vars.i_itr):
            delta_norm = delta().norm()
            self.avg['delta_norm'].update(delta_norm)
            self.add_to_hist({
                'i_itr': vars.i_itr,
                self.name: self.avg['delta_norm'].val(),
            })


class BatchPerSecondCb(BoardCallback):
    def __init__(self, n=100, **kwargs):
        super().__init__(**kwargs)
        self.prev = None
        self.sum = deque(maxlen=n)

    def on_batch_end(self, vars: StageVars):
        now = time.time()
        if self.prev is not None:
            rate = 1 / (now - self.prev)
            self.sum.append(1 / rate)
            s = np.array(self.sum)
            bps = len(s) / s.sum()
            self.add_to_hist({'batch_per_second': bps, 'i_itr': vars.i_itr})
        self.prev = now
        super().on_batch_end(vars)


class TimeElapsedCb(BoardCallback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._state['time_elapsed'] = 0

    def on_train_begin(self, vars: StageVars):
        self.start_time = time.time()
        self.offset = self.time_elapsed

    def on_batch_end(self, vars: StageVars):
        now = time.time()
        self.time_elapsed = int(now - self.start_time) + self.offset
        self.add_to_bar_and_hist({
            'i_itr': vars.i_itr,
            'time': self.time_elapsed
        })
        super().on_batch_end(vars)


class VisualizeWeightCb(BoardCallback):
    def on_batch_end(self, trainer, vars: StageVars):
        if self.should_write(vars.i_itr):
            param = nn.utils.parameters_to_vector(iter_opt_params(trainer.opt))
            self.writer.add_histogram('weight/hist/all', param, vars.i_itr)
            self.writer.add_scalar('weight/norm/all', param.norm(), vars.i_itr)


def list_name(net, prefix=''):
    """traverse the module to get each layer and their names,
    it outputs in the same order as net.parameters()"""
    has_child = False

    out = []
    for i, m in enumerate(net.children()):
        has_child = True
        m_name = f'{prefix}/{i}_{m.__class__.__name__}'
        out += list_name(m, m_name)

    if not has_child:
        for name, p in net.named_parameters():
            out.append((f'{prefix}/{name}', p))

    return out


class VisualizeWeightByLayerCb(BoardCallback):
    def on_batch_end(self, vars: StageVars):
        if self.should_write(vars.i_itr):
            for name, p in list_name(vars.trainer.net):
                # you will have the leading /
                self.writer.add_histogram(f'weight/hist{name}', p, vars.i_itr)
                self.writer.add_scalar(f'weight/norm{name}', p.norm(),
                                       vars.i_itr)
