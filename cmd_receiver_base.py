from abc import ABC, abstractmethod

class ACmdReceiver(ABC):
    @abstractmethod
    def receive(self) -> str: ...
