export CUDA_VISIBLE_DEVICES=2
./build/tools/caffe train --solver=models/bvlc_alexnet/solver_admm.prototxt --weights=models/bvlc_alexnet/caffe_alexnet_retrain_iter_400000.caffemodel

