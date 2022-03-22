from .cell import NCell, RCell, UCell, NCellNSkip, RCellNSkip, UCellNSkip, RCellNSkipOld,UCellNSkipOld, NCellNSkipOld


class NetConstructor:
    skip_cell_types = {'n':NCell,'r':RCell,'u':UCell}
    no_skip_cell_types = {'n':NCellNSkip,'r':RCellNSkip,'u':UCellNSkip}
    no_skip_old_cell_types = {'n':NCellNSkipOld,'r':RCellNSkipOld,'u':UCellNSkipOld}
    def __init__(self):
        pass
    @classmethod
    def construct(cls, cell_type, args, use_skip,old_ver=0):
        if old_ver:
            return cls.no_skip_old_cell_types[cell_type](*args)
        if use_skip:
            return cls.skip_cell_types[cell_type](*args)
        else:
            return cls.no_skip_cell_types[cell_type](*args)


if __name__ == '__main__':
    import torch
    a = NCell(20, 10, 10, 'n').cuda()
    b = RCell(20, 10, 10, 'n').cuda()
    c = UCell(20, 10, 10, 'n').cuda()
    inpit = torch.randn((1,10,32,32), device='cuda')
    print(a(inpit, inpit).shape)
    print(b(inpit, inpit).shape)
    print(c(inpit, inpit).shape)