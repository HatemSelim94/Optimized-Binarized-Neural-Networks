from .cell import NCell, RCell, UCell


class NetConstructor:
    cell_types = {'n':NCell,'r':RCell,'u':UCell}
    def __init__(self):
        pass
    @classmethod
    def construct(cls, celltype, args):
        return cls.cell_types[celltype](*args)


if __name__ == '__main__':
    import torch
    a = NCell(20, 10, 10, 'n').cuda()
    b = RCell(20, 10, 10, 'n').cuda()
    c = UCell(20, 10, 10, 'n').cuda()
    inpit = torch.randn((1,10,32,32), device='cuda')
    print(a(inpit, inpit).shape)
    print(b(inpit, inpit).shape)
    print(c(inpit, inpit).shape)