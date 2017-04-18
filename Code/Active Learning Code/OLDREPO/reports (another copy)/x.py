import matplotlib.pyplot as plt
import sys
from scipy.interpolate import spline
x = []
file_in = open('bla.da', 'r')
for y in file_in.read().split(' '):
	x.append(float(y))
	

#print x	
plt.plot(x)
plt.ylabel('fmeasure with '+sys.argv[1])
plt.xlabel('iterations')
plt.title(sys.argv[1])
plt.savefig('plots/'+sys.argv[1]+'.png')
#savefig()
#plt.show()
