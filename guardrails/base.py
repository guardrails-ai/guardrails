from abc import ABC, abstractmethod

class Callback(ABC):

    @abstractmethod
    def before_prepare(self, *args, **kwargs):
        ...

    @abstractmethod
    def after_prepare(self, *args, **kwargs):
        ...

    @abstractmethod
    def before_call(self, *args, **kwargs):
        ...

    @abstractmethod
    def after_call(self, *args, **kwargs):
        ...

    @abstractmethod
    def before_parse(self, *args, **kwargs):
        ...

    @abstractmethod
    def after_parse(self, *args, **kwargs):
        ...

    @abstractmethod
    def before_validate(self, *args, **kwargs):
        ...

    @abstractmethod
    def after_validate(self, *args, **kwargs):
        ...

    @abstractmethod
    def before_introspect(self, *args, **kwargs):
        ...

    @abstractmethod
    def after_introspect(self, *args, **kwargs):
        ...