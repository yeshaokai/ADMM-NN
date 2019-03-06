#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void SGDUpdate(int N, Dtype* g, Dtype* h,
    Dtype momentum, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    g[i] = h[i] = momentum*h[i] + local_rate*g[i];
  }
}
template <typename Dtype>
void sgd_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate) {
  SGDUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, h, momentum, local_rate);
  CUDA_POST_KERNEL_CHECK;
}
template void sgd_update_gpu<float>(int, float*, float*, float, float);
template void sgd_update_gpu<double>(int, double*, double*, double, double);

template <typename Dtype>
__global__ void abs_min_filter_kernel(int N, Dtype* a, Dtype min) {
  CUDA_KERNEL_LOOP(i, N) {
    if (abs(a[i]) < min) a[i] = 0;
  }
}
template <typename Dtype>
void abs_min_filter_gpu(int N, Dtype* a, Dtype min) {
  abs_min_filter_kernel<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, min);
  CUDA_POST_KERNEL_CHECK;
}
template void abs_min_filter_gpu<float>(int, float*, float);
template void abs_min_filter_gpu<double>(int, double*, double);

template <typename Dtype>
__global__ void set_mask_gpu_kernel(int N, const Dtype* a, Dtype min, Dtype* mask) {
  CUDA_KERNEL_LOOP(i, N) {
    if (abs(a[i]) < min)
      mask[i] = Dtype(0);
    else
      mask[i] = Dtype(1);
  }
}
template <typename Dtype>
void set_mask_gpu(int N, const Dtype* a, Dtype min, Dtype* mask) {
  set_mask_gpu_kernel<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, min, mask);
  CUDA_POST_KERNEL_CHECK;
}
template void set_mask_gpu<float>(int, const float*, float, float*);
template void set_mask_gpu<double>(int, const double*, double, double*);





template <typename Dtype>
__global__ void keep_pruned_weights_unchanged_gpu_kernel(int N, const Dtype* weight, Dtype * mask)
{
	CUDA_KERNEL_LOOP(i,N)
	{
		if (weight[i] == Dtype(0))
		{
			mask[i] = Dtype(0);
		}
		else
		{
			mask[i] = Dtype(1);
		}

	}
}

template <typename Dtype>
void keep_pruned_weights_unchanged_gpu(int N, const Dtype * weight, Dtype * mask)
{
	keep_pruned_weights_unchanged_gpu_kernel<Dtype>
	<<<CAFFE_GET_BLOCKS(N),CAFFE_CUDA_NUM_THREADS>>>(
	N,weight,mask);
	CUDA_POST_KERNEL_CHECK;
}

template void keep_pruned_weights_unchanged_gpu(int N, const float * weight, float * mask);
template void keep_pruned_weights_unchanged_gpu(int N, const double * weight, double * mask);

template <typename Dtype>
__global__ void replace_grads_by_centroids_gpu_kernel(int N, const Dtype* centroid_grads, const Dtype* labels, Dtype* grads)
{
	CUDA_KERNEL_LOOP(i,N)
	{
    int label = (int)labels[i];
    if (label >= 0) {
      grads[i] = centroid_grads[label];
    }
	}
}
template <typename Dtype>
void replace_grads_by_centroids_gpu(int N, const Dtype* centroid_grads, const Dtype* labels, Dtype* grads)
{
	replace_grads_by_centroids_gpu_kernel<Dtype>
	<<<CAFFE_GET_BLOCKS(N),CAFFE_CUDA_NUM_THREADS>>>(
	N, centroid_grads, labels, grads);
	CUDA_POST_KERNEL_CHECK;
}

template void replace_grads_by_centroids_gpu(int N, const float* centroid_grads, const float* labels, float* grads);
template void replace_grads_by_centroids_gpu(int N, const double* centroid_grads, const double* labels, double* grads);

template <typename Dtype>
__global__ void get_centroid_grad_gpu_kernel(int N, Dtype* centroid_grads, const Dtype* labels, Dtype* grads)
{
	CUDA_KERNEL_LOOP(i,N)
	{
    int label = (int)labels[i];
    if (label >= 0) {
      atomicAdd(centroid_grads + label, grads[i]);
    }
	}
}
template <typename Dtype>
void get_centroid_grad_gpu(int N, Dtype* centroid_grads, const Dtype* labels, Dtype* grads)
{
	get_centroid_grad_gpu_kernel<Dtype>
	<<<CAFFE_GET_BLOCKS(N),CAFFE_CUDA_NUM_THREADS>>>(
	N, centroid_grads, labels, grads);
	CUDA_POST_KERNEL_CHECK;
}

template void get_centroid_grad_gpu(int N, float* centroid_grads, const float* labels, float* grads);
template void get_centroid_grad_gpu(int N, double* centroid_grads, const double* labels, double* grads);
}  // namespace caffe

