#!/usr/bin/env sh
set -e
export CUDA_VISIBLE_DEVICES=0
# ADMM. Use pretrained model
./build/tools/caffe train --solver=examples/mnist/lenet_solver_admm_clustering.prototxt --weights=examples/mnist/lenet_pruning_iter_50000.caffemodel

# retrain
#./build/tools/caffe train --solver=examples/mnist/lenet_solver_retrain.prototxt --weights=examples/mnist/lenet_admm_lr01.caffemodel 
