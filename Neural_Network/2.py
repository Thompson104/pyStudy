import numpy as np
from matplotlib import pyplot as plt

fnx = lambda : np.random.randint(5, 50, 10)
y = np.row_stack((fnx(), fnx(), fnx()))
x = np.arange(10)

y1, y2, y3 = fnx(), fnx(), fnx()


p1 = plt.subplot(211)
#ax = fig.add_subplot(211)
p1.stackplot(x, y)



p2=plt.subplot(212)
p2.stackplot(x, y1, y2, y3)
plt.show()