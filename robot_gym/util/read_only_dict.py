from typing import TypeVar, Mapping, Dict

KT = TypeVar('KT')  # Key type.
VT = TypeVar('VT')  # Value type.


class ReadOnlyDict(Mapping[KT, VT]):
    """
    Defines a read-only wrapper around a built-in dictionary.
    """
    def __init__(self, inner_dict: Dict[KT, VT]):
        self.__inner_dict = inner_dict

    def __getitem__(self, item):
        return self.__inner_dict.__getitem__(item)

    def __iter__(self):
        return self.__inner_dict.__iter__()

    def __len__(self):
        return self.__inner_dict.__len__()
