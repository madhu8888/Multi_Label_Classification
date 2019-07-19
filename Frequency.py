import scipy.sparse as sparse
from pylab import *
import matplotlib.pyplot as plt
import matplotlib.pyplot as plot
import statistics
data = sparse.load_npz('Y.npz')

x_list=[]
data_list=[]

for i in range(3953):
    x_list.append(i)
    data_list=data[:, i].sum()
    # print(data_list)

x=[i for i in range(3953)]
y=[data[:, i].sum() for i in range(3953)]
y.sort()
print(statistics.mean(y))
print(len(y))
print(max(y))
print(y.count(4))

plt.plot(x, y, linestyle='-')


plot.xlabel('Labels')
plot.ylabel('Frequency')
plot.show()
