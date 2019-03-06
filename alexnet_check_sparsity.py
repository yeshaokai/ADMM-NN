import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
deploy_prototxt_path = './deploy.prototxt'
import sys
caffe_root = '/home/jiayu/caffe-admm-quantization1/'
sys.path.insert(0, caffe_root + 'python')
import caffe
caffemodel_path = '/data2/jiayu/gpu2/clustering_01/caffe_alexnet_train_admm_iter_1500000.caffemodel'
net = caffe.Net(deploy_prototxt_path,caffemodel_path,caffe.TEST)
for k in net.params.keys():
    print ('at layer %s' % k)
    array =net.params[k][0].data
    print ('unique number of weights %d' % len(np.unique(array)))
    plt.axis([-0.05, 0.05, 0, 100000])
    _ = plt.hist(array.flatten(), bins=100, normed=0, facecolor='blue')
    print("start print")
    #plt.show()
    plt.savefig("/home/jiayu/caffe-admm-quantization1/"+str(k)+".png")
    print("finish print")
