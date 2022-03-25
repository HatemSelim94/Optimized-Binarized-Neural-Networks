from .cell import NCellNSkipOld, RCellNSkipOld, UCellNSkipOld


class NetConstructor:
    no_skip_old_cell_types = {'n':NCellNSkipOld,'r':RCellNSkipOld,'u':UCellNSkipOld}
    def __init__(self):
        pass
    @classmethod
    def construct(cls, cell_type, args, use_skip, old_ver=0):
        if old_ver:
            return cls.no_skip_old_cell_types[cell_type](*args)
