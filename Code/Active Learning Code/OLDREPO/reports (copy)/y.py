import matplotlib.pyplot as plt
import sys
from scipy.interpolate import spline
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('blas') if isfile(join('blas', f))]
for fileda in onlyfiles:
	print fileda;
	x = []
	file_in = open('blas/'+fileda, 'r')
	for y in file_in.read().split(' '):
		x.append(float(y))
		

	#print x	
	plt.plot(x)
	plt.ylabel('fmeasure ')
	plt.xlabel('iterations')
	plt.title("non")

plt.show()
#plt.savefig('plots/'+sys.argv[1]+'.png')

#savefig()
#plt.show()
