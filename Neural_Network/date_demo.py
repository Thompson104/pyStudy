import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

years = mdates.YearLocator()
months = mdates.MonthLocator()
yearsFmt = mdates.DateFormatter('%Y')

datafile = cbook.get_sample_data('goog.npy')
r = np.load(datafile).view(np.recarray)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(r.date,r.adj_close)

ax.xaxis.set_major_locator(years)

plt.show()