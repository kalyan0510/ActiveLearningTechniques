# USAGE
# python lenet_mnist.py --save-model 1 --weights output/lenet_weights.hdf5
# python lenet_mnist.py --load-model 1 --weights output/lenet_weights.hdf5

# import the necessary packages
import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32,cuda.root=/usr/local/cuda-8.0/bin"
import numpy as np
import argparse
import cv2
import theano
import theano.tensor as T
import cPickle as pickle
import gzip
import sys
def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))


# Print iterations progress
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100, fill = '\xe2'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(barLength * iteration // total)
    bar = fill * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def dispImg(imarray,sz,dsz,loc):
	e = np.array( imarray ).reshape(-1,sz);
	image =  (e).astype("uint8")
	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (dsz, dsz), interpolation=cv2.INTER_LINEAR)
	cv2.putText(image, loc,(5, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
	cv2.imshow("image", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def getArr(model,imarray,sz):
	z = imarray.reshape(sz,sz)
	z = blockshaped(z,20,20)
	#print z.shape
	z = z[:,:,:,np.newaxis]
	#z = z.reshape(sz*sz/400,400)
	res = []
	x = 0
	for i in range(z.shape[0]):
		probs = model.predict(z[np.newaxis, i])
		prediction = probs.argmax(axis=1)
		res.append(prediction[0])
	x = np.array(res).reshape(sz/20,sz/20).sum(axis=1).argmax()
	if(np.array(res).reshape(sz/20,sz/20).sum(axis=1).max() == 0):
		return 0
	return x*20 + 10

def load_data_shared(filename="data/mnist.pkl.gz"):
	f = gzip.open(filename, 'rb')
	training_data = pickle.load(f)
	f.close()
	return training_data

print 'loading data(750MB)'
training_data = load_data_shared("data/data1005.pkl.gz")

td , tl , ts, tw = training_data
#td = td.get_value();
sz=20;
f = open("test1.arff",'w');
f.write("@RELATION testing\n");
for j in range(td.shape[1]):
	f.write("@ATTRIBUTE "+"v"+str((j+1))+" numeric\n");

f.write("@ATTRIBUTE class {0,1}\n");
f.write("@DATA\n");

for i in range(td.shape[0]):
	for j in range(td.shape[1]):
		f.write(str('{0:f}'.format(td[i][j])));
		f.write(',');
	f.write(str(tl[i]));
	f.write('\n');
	

f.close();


	


