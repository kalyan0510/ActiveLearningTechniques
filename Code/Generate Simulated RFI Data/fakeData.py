from __future__ import print_function

import sys, random, os, pickle, gzip

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def makePixel(binary):
	return (1 if random.random() > 0.5 else 0) if binary else random.random()

def rfiCurtain(img_name, op_path, size_x, size_y, binary):
	rfi_axis = 'x' #if random.random()>0.5 else 'y'
	if rfi_axis == 'x':
		rfi_start = random.randint(size_x/10,size_x - (size_x/10))
		rfi_width = (int(size_x/(25*random.randint(1,size_x)))  if int(size_x/(25*random.randint(1,size_x)))>1 else int(size_x/(25*random.randint(1,size_x))))+(1) #CHANG
	else:
		rfi_start = random.randint(size_y/10,size_y - (size_y/10))
		rfi_width = int(size_y/(10*random.randint(1,size_y)))
	rfi_width = random.randint(1,3)	
	img = [0 for j in range(size_x*size_y)]
	for i in range(size_y):
		for j in range(size_x):
			intensity = makePixel(binary)
			if ((rfi_axis=='y') and (j>=rfi_start and j<=rfi_start+rfi_width)) or ((rfi_axis=='x') and (i>=rfi_start and i<=rfi_start+rfi_width)):
				bais = 0.1 if ((rfi_axis=='y') and (j<=rfi_start+int(rfi_width*0.1) or j>=rfi_start+int(rfi_width*0.1))) or ((rfi_axis=='x') and (i<=rfi_start+int(rfi_width*0.1) and i>=rfi_start+int(rfi_width*0.9))) else 0.0
				intensity = 0 if random.random() < 0.01+bais else 1				
			img[i*size_x+j] = intensity
	plotImg(img, op_path+rfi_axis+'_'+img_name+'.png')
	if rfi_axis=='x':
		return (img, np.int64(1),np.int64(rfi_start),np.int64(rfi_width))
	return (img, np.int64(3))

def pulsarCurtain(img_name, op_path, size_x, size_y, binary):
	rfi_start_x = random.randint(1,size_x-1)
	rfi_end_x =  random.randint(1+rfi_start_x,size_x)
	rfi_start_y = random.randint(1,size_y-1)
	rfi_end_y = random.randint(1+rfi_start_y,size_y)
	rfi_width = random.randint(1,3)
	img = [0 for j in range(size_x*size_y)]
	c_1 = (((rfi_start_x+rfi_end_x)/2.0)*((rfi_start_y+rfi_end_y)/2.0))
	c_2 = (((rfi_start_x+rfi_end_x)/2.0+rfi_width*0.1)*((rfi_start_y+rfi_end_y)/2.0+rfi_width*0.1))
	c_3 = (((rfi_start_x+rfi_end_x)/2.0+rfi_width*0.9)*((rfi_start_y+rfi_end_y)/2.0+rfi_width*0.9))
	c_4 = (((rfi_start_x+rfi_end_x)/2.0+rfi_width)*((rfi_start_y+rfi_end_y)/2.0+rfi_width))
	for i in range(size_y):
		for j in range(size_x):
			intensity = makePixel(binary)
			if ((i>=rfi_start_y and i<=rfi_end_y and j>=rfi_start_x and j<=rfi_end_x) and (i*j>c_1 and i*j<c_4)):
				coin = random.random()
				bais = 0.1 if ((i*j>c_1 and i*j<c_2) or (i*j>c_3 and i*j<c_4)) else 0.0
				intensity = 0 if coin < 0.01+bais else 1				
			img[i*size_x+j] = intensity
	plotImg(img, op_path+'pulsar'+img_name+'.png')
	return (img, np.int64(2), np.int64(0), np.int64(0))

def plotImg(imgsrc, imgName=''):
	'''
	img = [[0 for j in range(size_x)] for i in range(size_y)]
	for i in range(size_y):
		for j in range(size_x):
			img[i][j] = imgsrc[i*size_x+j]
	H = np.matrix(img)
	fig = plt.figure(frameon=False)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)
	plt.imshow(H, interpolation='none', aspect='auto')
	plt.savefig(imgName)
	'''

def generateRandomData(img_name, op_path, size_x, size_y, binary):
	if(random.random()>0.5):
		return rfiCurtain(img_name, op_path, size_x, size_y, binary)
	else:
		return pulsarCurtain(img_name, op_path, size_x, size_y, binary)

def generateNormalData(img_name, op_path, size_x, size_y, binary):
	i = 0
	j = 0
	img = [np.float32(0.0) for i in range(size_x*size_y)]
	for i in range(size_y*size_x):
		intensity = makePixel(binary)		
		img[i] = intensity
	plotImg(img, op_path+img_name+'.png')
	return (img, np.int64(0), np.int64(0), np.int64(0))

def compressData(samples):
	training_data_x = []
	validation_data_x = []
	test_data_x = []
	training_data_y = []
	validation_data_y = []
	test_data_y = []
	training_data_s = []
	training_data_w = []
	j = 0 # counter
	num_samples = len(samples)
	random.shuffle(samples)
	random.shuffle(samples)
	#print(type(samples[0][0].item(0)))
	for data in samples:
		training_data_x.append(data[0])
		training_data_y.append(data[1])
		training_data_s.append(data[2])
		training_data_w.append(data[3])
		j+=1
	
	#validation_data = (np.array(validation_data_x), np.array(validation_data_y))
	#test_data = (np.array(test_data_x), np.array(test_data_y))
	training_data = (np.array(training_data_x), np.array(training_data_y),np.array(training_data_s),np.array(training_data_w))
	print("Saving data. This may take a few minutes.")
	f = gzip.open("data/data1005.pkl.gz", "w")
	pickle.dump((training_data), f)
	f.close()

if __name__ == "__main__":
	if len(sys.argv)!=6:
		print ("Usage: fakeData.py op_path size_x size_y binary num_images")
		exit(-1)

	op_path = sys.argv[1]+"/"
	size_x = int(sys.argv[2])
	size_y = int(sys.argv[3])	
	binary = int(sys.argv[4])
	num_images = int(sys.argv[5])
	
	samples = []
	for imgcnt in range(num_images):
		img_name = "img_"+str(imgcnt)
		samples.append(rfiCurtain(img_name, op_path, size_x, size_y, binary))
		samples.append(pulsarCurtain(img_name, op_path, size_x, size_y, binary))
		samples.append(generateNormalData(img_name, op_path, size_x, size_y, binary))

	compressData(samples)
