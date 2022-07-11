from abc import ABC, abstractmethod
import numpy as np

class Digit:

    def __init__(self, width, height, channels, fps):
        self.__fps = fps
        self.__width = width
        self.__height = height
        self.__channels = channels

    @abstractmethod
    def get_frame(self) -> np.ndarray:
        pass

    @property
    def frame(self):
        return self.get_frame()

    @property
    def fps(self) -> int:
        return self.__fps

    @property
    def width(self) -> int:
        return self.__width

    @property
    def height(self) -> int:
        return self.__height

    @property
    def channels(self) -> int:
        return self.__channels