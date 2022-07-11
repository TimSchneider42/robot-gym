from collections import defaultdict
from threading import Lock
from typing import TypeVar, Callable, Optional


class SynchronizedObserver:
    def start_observing(self):
        pass

    def finish_observing(self):
        pass


class SimpleSynchronizedObserver(SynchronizedObserver):
    def __init__(self, observation_lock: Optional[Lock] = None):
        self.__observed_functions = {
            el._sync_obs_inner_func.__name__: el for el in [getattr(self.__class__, e) for e in dir(self.__class__)]
            if callable(el) and hasattr(el, "_sync_obs_inner_func")
        }
        self._observed_values = defaultdict(lambda: None)
        self.__observation_lock = observation_lock

    def finish_observing(self):
        if self.__observation_lock is not None:
            with self.__observation_lock:
                self.__observe()
        else:
            self.__observe()

    def __observe(self):
        for fn, f in self.__observed_functions.items():
            self._observed_values[fn] = f._sync_obs_inner_func(self)


T = TypeVar("T")


def synchronized_obs(func: Callable[[], T]) -> Callable[[], T]:
    def wrapper(*args, _inner_func=func, **kwargs) -> T:
        return args[0]._observed_values[_inner_func.__name__]

    wrapper._sync_obs_inner_func = func

    return wrapper
