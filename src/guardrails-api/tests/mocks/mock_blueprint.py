from typing import List


class MockBlueprint:
    name: str
    module_name: str
    routes: List[str]
    methods: List[str]
    route_call_count: int

    def __init__(self, name: str, module_name: str, **kwargs):
        self.name = name
        self.module_name = module_name
        self.routes = []
        self.methods = []
        self.route_call_count = 0
        for key in kwargs:
            self.__setitem__(key, kwargs.get(key))

    def route(self, route_name: str, methods: List[str] = []):
        def no_op(fn, *args):
            return fn

        self.routes.append(route_name)
        self.methods.extend(methods)
        unique_methods = list(set(self.methods))
        self.methods = unique_methods
        self.route_call_count = self.route_call_count + 1
        return no_op

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        delattr(self, key)
