import time
from collections import deque
from itertools import chain
from queue import Queue
from threading import Lock
from typing import TypeVar, Optional, NamedTuple, Callable, Tuple, Dict, Any


class SynchronizedActor:
    def start_acting(self):
        pass

    def finish_acting(self):
        pass


ActionProtocolEntry = NamedTuple(
    "ActionProtocolEntry", (("time", float), ("function", Callable), ("args", Tuple), ("kwargs", Dict[str, Any])))


class SimpleSynchronizedActor(SynchronizedActor):
    def __init__(self, action_lock: Optional[Lock] = None, action_protocol_length: int = 0):
        self._act_queue = Queue()
        self.__action_lock = action_lock
        self.__action_protocol = deque(maxlen=action_protocol_length)
        self.__action_protocol_lock = Lock()

    def finish_acting(self):
        if self.__action_lock is not None:
            with self.__action_lock:
                self.__act()
        else:
            self.__act()

    def __act(self):
        with self.__action_protocol_lock:
            while not self._act_queue.empty():
                f, args, kwargs = self._act_queue.get()
                self.__action_protocol.append(ActionProtocolEntry(time.time(), f, args, kwargs))
                f(*args, **kwargs)

    @property
    def action_protocol(self) -> Tuple[ActionProtocolEntry, ...]:
        return tuple(self.__action_protocol)

    def get_action_protocol_str(self) -> str:
        with self.__action_protocol_lock:
            protocol_copy = list(self.__action_protocol)
        lines = [
            "[{}] {}({})".format(t, fun.__name__, ", ".join(
                chain(map(str, args), ("{}={}".format(k, v) for k, v in kwargs.items()))))
            for t, fun, args, kwargs in protocol_copy]
        return "\n".join(lines)


T = TypeVar("T")


def synchronized_act(func):
    def wrapper(*args, _inner_func=func, **kwargs):
        args[0]._act_queue.put((_inner_func, args, kwargs))

    return wrapper
