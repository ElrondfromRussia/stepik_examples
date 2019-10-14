class DoubElIter:
    def __init__(self, lst):
        self.lst = lst
        self.i = 0

    def __next__(self):
        if self.i < len(self.lst):
            self.i += 2
            return self.lst[self.i - 2], self.lst[self.i - 1]
        else:
            raise StopIteration


class MList(list):
    def __iter__(self):
        return DoubElIter(self)


x = MList()
x.extend([1, 2, 3, 4, 5, 6])
for i in x:
    print(i)
