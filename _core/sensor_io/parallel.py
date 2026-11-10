"""Bounded, spawn-safe record parallelism. Numerical arrays stay in workers."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import multiprocessing
import os


def initialize_worker(threads):
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key]=str(threads)
    # NumPy may already have loaded a numerical runtime in the spawned interpreter.
    global _thread_limit
    from threadpoolctl import threadpool_limits
    _thread_limit=threadpool_limits(limits=threads)


@contextmanager
def worker_environment(threads):
    keys=("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")
    previous={key:os.environ.get(key) for key in keys}
    try:
        for key in keys:os.environ[key]=str(threads)
        yield
    finally:
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


def record_results(function,tasks,workers=1,threads=1):
    """Keep at most workers futures pending; failures stop new record dispatch."""
    if workers<1 or threads<1:raise ValueError("Worker and numerical thread counts must be positive")
    if workers==1:
        from threadpoolctl import threadpool_limits
        scope=threadpool_limits(limits=threads)
        with scope:
            for task in tasks:yield function(task)
        return
    with worker_environment(threads):
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize_worker,initargs=(threads,)) as pool:
            iterator=iter(tasks);pending=set()
            for _ in range(workers):
                try:pending.add(pool.submit(function,next(iterator)))
                except StopIteration:break
            try:
                while pending:
                    done,pending=wait(pending,return_when=FIRST_COMPLETED)
                    completed=[future.result() for future in done]
                    for value in completed:yield value
                    for _ in completed:
                        try:pending.add(pool.submit(function,next(iterator)))
                        except StopIteration:break
            except BaseException:
                for future in pending:future.cancel()
                raise
