from multiprocessing.dummy import Pool
from multiprocessing import get_context


def multiprocess_map(fn, args, num_workers: int):
    with Pool(num_workers) as pool:
        pool.map(_call_fn_under_process, [(fn, arg) for arg in args])


def _call_fn_under_process(fn_arg):
    fn, arg = fn_arg
    ctx = get_context('spawn')
    proc = ctx.Process(target=fn, args=(arg, ))
    proc.start()
    proc.join()

def _test(arg):
    print('arg:', arg)

if __name__ == '__main__':
    multiprocess_map(_test, [1, 2, 3, 4], 2)
