class ListPlusPlus(list):
    def __init__(self, *args):
        list.__init__(self, args)

    def at(self, index: int):
        value = None
        try:
            value = self[index]
        except IndexError:
            pass
        return value
