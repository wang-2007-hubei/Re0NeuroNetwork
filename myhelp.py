import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
x,y=spiral_data(100,3)
print(x)
print(y)