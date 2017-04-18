import matplotlib.pyplot as plt
import sys
from scipy.interpolate import spline
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('blas') if isfile(join('blas', f))]
val=0
x = []
fileda = onlyfiles[1]
file_in = open('blas/'+fileda, 'r')
for y in file_in.read().split(' '):
	x.append(float(y))
	
#print x
an=fileda.split('.')
plt.plot(x, label=an[0] )
plt.ylabel('fmeasure ')
plt.xlabel('iterations')
plt.title("plot")

#plt.ylim(0,1) 
plt.legend(bbox_to_anchor=(0.85, 0.7), loc=1, borderaxespad=0.)
plt.show()
#plt.savefig('plots/'+'all'+'.png')

#savefig()
#plt.show()
