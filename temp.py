
class test():

    def __init__(self, data=1):
        self.data = data

    def __iter__(self):
        while True:
            if self.data < 100:
                self.data += 1
                yield self.data
            else:
                break

    
gene = test(10)
gene_iter = iter(gene)
flag = True

for i in range(10):
    item = next(gene_iter)
    print(item)
    if i > 5 and flag:
        gene = test(50)
        gene_iter = iter(gene)
        flag = False