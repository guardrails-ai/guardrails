from typing import Callable

class Callback:
    def __init__(self, before_prepare: Callable = None,
        after_prepare: Callable = None, before_call: Callable = None):
        self.before_prepare = before_prepare
        self.after_prepare = after_prepare
        self.before_call = before_call

    def before_prepare(self, **kwargs):
        if self.before_prepare is not None:
            self.before_prepare(**kwargs)

    def after_prepare(self, **kwargs):
        if self.before_prepare is not None:
            self.after_prepare(**kwargs)

    def before_call(self, **kwargs):
        if self.before_prepare is not None:
            self.before_call(**kwargs)

