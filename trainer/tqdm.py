"""
mulit-processing tqdm support
tqdm cannot be run with multiple processes, unless its outputs would collide.
we create a central process to manage all the outputs for tqdm. 
each subprocess would get a "surrogate" which signals to the central to update.
"""
from tqdm.autonotebook import tqdm
import multiprocessing as mp
import threading as th

DEFAULT_TQDM = None

def has_default_tqdm():
    return DEFAULT_TQDM is not None

def get_default_tqdm():
    return DEFAULT_TQDM

def set_default_tqdm(obj):
    global DEFAULT_TQDM
    DEFAULT_TQDM = obj

class Surrogate:
    """tqdm surrogate object it should look like tqdm.
    this object is put to the process where signals are redirected back the main process."""
    def __init__(self, remote: mp.connection.Connection):
        self.remote = remote
        self.n = 0
        self.total = None
        self.desc = ''

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        self.remote.close()  # close pipes

    def update(self, n: int = 1):
        """set n of the underlying tdqm"""
        self.n += n
        self.remote.send(('update', n))

    def set_description(self, desc, **kwargs):
        self.desc = desc
        self.remote.send(('set_description', {'args': (desc, ), 'kwargs': kwargs}))

    def set_postfix(self, *args, **kwargs):
        self.remote.send(('set_postfix', {'args': args, 'kwargs': kwargs}))

    def reset(self, total=None):
        self.n = 0
        self.remote.send(('reset', total))
        self.total = total

    def close(self):
        return self.remote.close()

def _synchronizer(remote, progress: tqdm):
    while True:
        try:
            cmd, arg = remote.recv()
            if cmd == 'update':
                progress.update(arg)
            elif cmd == 'set_description':
                progress.set_description(*arg['args'], **arg['kwargs'])
            elif cmd == 'reset':
                progress.reset(arg)
            elif cmd == 'set_postfix':
                # must not refresh (which will make the tqdm refreshes too frequently)
                progress.set_postfix(*arg['args'], **arg['kwargs'])
            else:
                raise NotImplementedError(f'cmd not found {cmd}')
        except KeyboardInterrupt:
            break
        except EOFError:  # pipe closed
            break

def create_surrogate(progress: tqdm):
    """tqdm is not multi-process safe. 
    we need to make all updates done in the main process.
    we create a surrogate object for a process, 
    which all updates are redirected to the main process.
    """
    here, there = mp.Pipe()
    surrogate = Surrogate(there)
    thread = th.Thread(
        target=_synchronizer, kwargs={
            'remote': here,
            'progress': progress,
        }
    )
    thread.daemon = True
    thread.start()
    return surrogate