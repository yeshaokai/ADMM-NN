#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include "caffe/sgd_solvers.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"
//modify here
int cmp(const std::pair<int, float>& x, const std::pair<int, float>& y)
{
  return x.second < y.second;
}    
namespace caffe {

//modify here
template<typename Dtype>
void SGDSolver<Dtype>::sortMapByValue(map<int, float>& tMap,vector<std::pair<int, float> >& tVector)
{
    for (map<int, float>::iterator curr = tMap.begin(); curr != tMap.end(); curr++)
      tVector.push_back(std::make_pair(curr->first, curr->second));

    std::sort(tVector.begin(), tVector.end(), cmp);
 }    
// Return the current learning rate. The currently implemented learning rate
// policies are as follows:
//    - fixed: always return base_lr.
//    - step: return base_lr * gamma ^ (floor(iter / step))
//    - exp: return base_lr * gamma ^ iter
//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
//    - multistep: similar to step but it allows non uniform steps defined by
//      stepvalue
//    - poly: the effective learning rate follows a polynomial decay, to be
//      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
//    - sigmoid: the effective learning rate follows a sigmod decay
//      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
//
// where base_lr, max_iter, gamma, step, stepvalue and power are defined
// in the solver parameter protocol buffer, and iter is the current iteration.
template <typename Dtype>
Dtype SGDSolver<Dtype>::GetLearningRate() {
  Dtype rate;
  const string& lr_policy = this->param_.lr_policy();
  if (lr_policy == "fixed") {
    rate = this->param_.base_lr();
  } else if (lr_policy == "step") {
    this->current_step_ = this->iter_ / this->param_.stepsize();
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "exp") {
    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
  } else if (lr_policy == "inv") {
    rate = this->param_.base_lr() *
        pow(Dtype(1) + this->param_.gamma() * this->iter_,
            - this->param_.power());
  } else if (lr_policy == "multistep") {
    if (this->current_step_ < this->param_.stepvalue_size() &&
          this->iter_ >= this->param_.stepvalue(this->current_step_)) {
      this->current_step_++;
      LOG(INFO) << "MultiStep Status: Iteration " <<
      this->iter_ << ", step = " << this->current_step_;
    }
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "poly") {
    rate = this->param_.base_lr() * pow(Dtype(1.) -
        (Dtype(this->iter_) / Dtype(this->param_.max_iter())),
        this->param_.power());
  } else if (lr_policy == "sigmoid") {
    rate = this->param_.base_lr() * (Dtype(1.) /
        (Dtype(1.) + exp(-this->param_.gamma() * (Dtype(this->iter_) -
          Dtype(this->param_.stepsize())))));
  } else if (lr_policy == "admm") {  // for admm pruning algorithm
    int iter = this->iter_ % this->param_.admm_iter();
    //std::cout << "using admm lr" << std::endl;
    if (iter == 0) {
      this->current_step_ = 0;
      LOG(INFO) << "ADMM MultiStep Status: Iteration " <<
      iter << ", step = " << this->current_step_;
    } else if (this->current_step_ < this->param_.stepvalue_size() &&
        iter >= this->param_.stepvalue(this->current_step_)) {
      this->current_step_++;
      LOG(INFO) << "ADMM MultiStep Status: Inner Iteration " <<
      iter << ", step = " << this->current_step_;
    }
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else {
    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }
  return rate;
}

template <typename Dtype>
void SGDSolver<Dtype>::PreSolve() {
  // Initialize the history
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  history_.clear();
  update_.clear();
  temp_.clear();
  wtemp_.clear();
  zutemp_.clear();
  mask_.clear();
  //labelMatrix_.clear();
  //centroid_grads_.clear();
  centroids_.clear();
  centroids_ranges_.clear();
  int num_layers = this->net_->learnable_params().size();
  centroids_.resize(num_layers);
  centroids_ranges_.resize(num_layers);
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    update_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    temp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    wtemp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    zutemp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    mask_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    if (!this->net_->has_params_clustering_num()[i]){
      vector<int> num_cluster_shape_zero;
      num_cluster_shape_zero.push_back(1);
      centroids_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>(num_cluster_shape_zero));
      centroids_ranges_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>(num_cluster_shape_zero));
    }
    else{
    int num_clusters = this->net_->params_clustering_num()[i];
    std::cout<< i << "layer" << num_clusters<<std::endl;
    vector<int> num_cluster_shape;
    num_cluster_shape.push_back(2 * num_clusters);
    vector<int> num_range_shape;
    num_range_shape.push_back(2 * num_clusters - 1); 
    centroids_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>(num_cluster_shape));
    centroids_ranges_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>(num_range_shape));
    }
  }
  //for (int param_id = 0; param_id < this->net_->learnable_params().size();++param_id)
  // {
    //if (!this->net_->has_params_clustering_num()[param_id])
    //		continue;
  // 	int num_clusters = this->net_->params_clustering_num()[param_id];
  //      vector<int> num_cluster_shape;
  //	num_cluster_shape.push_back(2 * num_clusters);
  // 	vector<int> num_range_shape;
  //	num_range_shape.push_back(2 * num_clusters - 1);
  // 	centroids_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(num_cluster_shape)));
  //	centroids_ranges_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(num_range_shape)));
  //}
}

template <typename Dtype>
void SGDSolver<Dtype>::ClipGradients() {
  const Dtype clip_gradients = this->param_.clip_gradients();
  if (clip_gradients < 0) { return; }
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  Dtype sumsq_diff = 0;
  for (int i = 0; i < net_params.size(); ++i) {
    sumsq_diff += net_params[i]->sumsq_diff();
  }
  const Dtype l2norm_diff = std::sqrt(sumsq_diff);
  if (l2norm_diff > clip_gradients) {
    Dtype scale_factor = clip_gradients / l2norm_diff;
    LOG(INFO) << "Gradient clipping: scaling down gradients (L2 norm "
        << l2norm_diff << " > " << clip_gradients << ") "
        << "by scale factor " << scale_factor;
    for (int i = 0; i < net_params.size(); ++i) {
      net_params[i]->scale_diff(scale_factor);
    }
  }
}

#ifndef CPU_ONLY
template <typename Dtype>
void set_mask_gpu(int N, const Dtype* a, Dtype min, Dtype* mask);
#endif

#ifndef CPU_ONLY
template <typename Dtype>
void keep_pruned_weights_unchanged_gpu(int N,const Dtype* a, Dtype * mask);
#endif




//quantization utility functions from here
//find max will be used to find abs value in weight matrix
template <typename Dtype>
float SGDSolver<Dtype>:: Find_max(const Dtype* weight, int param_id)
{
	const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
	int count = net_params[param_id]->count();
	float max_var = -1*std::numeric_limits<float>::max();
	for (int i = 0; i< count; i++)
	{
	  if (std::abs<float>(weight[i]) > max_var){
	    max_var = std::abs<float>(weight[i]);
		}
	}
	return max_var;
}
//quantization for getting assement points
template <typename Dtype>
vector<float> SGDSolver<Dtype>:: Get_assement_points(float s_start, float s_end, float assement_point, float step)
{
	vector<float> current_points;
    float new_step = step * assement_point;
    int num = (int)((s_end - s_start) / new_step);
    current_points.resize(num + 1);
    int index = 0;
    while (num >= 0)
    {
        current_points[index] = s_start + index * new_step;
        index = index + 1;
        num = num - 1;
    }
    return current_points;
}

//quantization for getting centroids
template <typename Dtype>
vector<float> SGDSolver<Dtype>:: Get_centroids(float scale, int one_side_clusters)
{
	vector<float> current_centroids;
	int start = one_side_clusters * (-1);
	int end = one_side_clusters;
	for (int i = start; i <= end; i++)
	{
	        if (i == 0)
		  continue;
		float temp = i * scale;
		current_centroids.push_back(temp);
	}
	return current_centroids;
}

//quantization for getting centroids range
template <typename Dtype>
vector<float> SGDSolver<Dtype>::Get_centroids_range(vector<float>& centroids)
{
	vector<float> c_range;
	for (int i = 0; i < centroids.size() - 1; ++i)
	{
		float two_sum = (centroids[i] + centroids[i + 1]) / 2;
        c_range.push_back(two_sum);
	}
	return c_range;
}

//quantization for getting calculate_square_error
template <typename Dtype>
float SGDSolver<Dtype>::Calculate_square_error(int param_id, vector<float>& cur_centroids, vector<float>& c_range)
{
	float total_error = 0;
	const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
	const Dtype* weight = net_params[param_id]->cpu_data();
	int count = net_params[param_id]->count();
	for (int i = 0; i < c_range.size(); ++i)
	{
		if(i == 0)
		{
			float diff1 = 0;
			for (int j = 0; j < count; ++j)
			{
				if (weight[j] < c_range[i])
				{
				  diff1 = diff1 + std::pow((weight[j] - cur_centroids[i]), 2);
				}
			}
			total_error = total_error + diff1;
		}

		else if(i == c_range.size() - 1)
		{
			float diff2 = 0;
			float diff3 = 0;
			for (int j = 0; j < count; ++j)
			{
				if (weight[j] >= c_range[i - 1] && weight[j] < c_range[i])
				{
				  diff2 = diff2 + std::pow((weight[j] - cur_centroids[i]),2);
				}
				else if (weight[j] >= c_range[i])
				{
				  diff3 = diff3 + std::pow((weight[j] - cur_centroids[i + 1]),2);
				}
			}
			total_error = total_error + diff2;
			total_error = total_error + diff3;

		}

		else if(i > 0 && i < c_range.size() - 1)
		{
			float diff4 = 0;
			for (int j = 0; j < count; ++j)
			{
				if (weight[j] >= c_range[i - 1] && weight[j] < c_range[i])
				{
				  diff4 = diff4 + std::pow((weight[j] - cur_centroids[i]),2);
				}
			}
			total_error = total_error + diff4;
		}	
	}
	return total_error;
}
template<typename Dtype>
void SGDSolver<Dtype>::Project_to_centorid(int param_id, Dtype* weight, const Dtype* cur_centroids, const Dtype* c_range)
{
	const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
	int count = net_params[param_id]->count();
	//int num_clusters = this->net_->params_clustering_num()[param_id];
	int range_count = centroids_ranges_[param_id]->count();
	std::cout<< "range_count: " << range_count << std::endl;
	for (int i = 0; i < range_count; ++i)
	{
	        std::cout<< "cur range: " << c_range[i] << std::endl;
		if (i == 0)
		{
			for (int j = 0; j < count; ++j)
			{
			        float zero = 0;
			        if (weight[j] == zero)
				  continue;
				if (weight[j] < c_range[i])
				{
					weight[j] = cur_centroids[i];
				}
			}
		}

		else if (i == (range_count - 1))
		{
			for (int j = 0; j < count; ++j)
			{
			        float zero = 0;
			        if (weight[j] == zero)
			            continue;
				if (weight[j] >= c_range[i - 1] && weight[j] < c_range[i])
				{
					weight[j] = cur_centroids[i];
				}
				else if (weight[j] >= c_range[i])
				{
					weight[j] = cur_centroids[i + 1];
				}
			}
		}

		else if (i > 0 && (i < range_count - 1))
		{
			for (int j = 0; j < count; ++j)
			{
			        float zero = 0;
			        if (weight[j] == zero)
			              continue;
				if (weight[j] >= c_range[i - 1] && weight[j] < c_range[i])
				{
					weight[j] = cur_centroids[i];
				}
			}
		}
	}

}
//modify here
template<typename Dtype>
bool SGDSolver<Dtype>::is_weight_quant_zero(int centroid_index, float cur_centroids, float weight, const Dtype * cur_centroids_ranges, int centroids_ranges_count)
{
    //int centroids_count = centroids_[param_id]->count();
    //centroids_ranges_[];
    float zero = 0;
    if((cur_centroids == weight) || (weight == zero))
      return true;

    if (centroid_index == 0)
      {
	if (weight < cur_centroids_ranges[centroid_index])
	  return false;
	else
	  return true;
      }
    else if(centroid_index > 0 && centroid_index < centroids_ranges_count)
      {
	if (weight >= cur_centroids_ranges[centroid_index - 1] && weight < cur_centroids_ranges[centroid_index])
	  return false;
	else
	  return true;
      }
    else if (centroid_index == centroids_ranges_count)
      {
	if (weight >= cur_centroids_ranges[centroid_index - 1])
	  return false;
	else
	  return true;
      }
    else
      {
	std::cout<<"out of check range"<< std::endl;
	return false;
      }
}


  //modify here
  //quantization for projecting threshold
  template<typename Dtype>
  void SGDSolver<Dtype>::Project_threshold(int param_id, Dtype* weight, Dtype* mask, const Dtype* cur_centroids, const Dtype * cur_centroids_ranges)
  {
    int modify = 0;
    int quantized_modify = 0;
    const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
    int count = net_params[param_id]->count();
    int centroids_count = centroids_[param_id]->count();
    std::cout<< "centroids_count: " << centroids_count << std::endl;
    int centroids_ranges_count = centroids_ranges_[param_id]->count();
    std::cout<< "centroids_ranges_count: " << centroids_ranges_count << std::endl;

    for (int i = 0; i < centroids_count; ++i)
      {
	//std::cout<< "weight: " << weight[i] << std::endl;
	std::map<int, float> weight_map;
	vector<std::pair<int, float> > tVector;
	for (int j = 0; j < count; ++j)
	  {
	    if (cur_centroids[i] == weight[j])
	      {
		mask[j] = Dtype(0);
		quantized_modify++;
	      }
	    if(is_weight_quant_zero(i, cur_centroids[i], weight[j], cur_centroids_ranges, centroids_ranges_count))
	      continue;
	    else
	      {
		float diff = std::abs<float>(weight[j] - cur_centroids[i]);
		  weight_map.insert(std::pair<int, float>(j, diff));
	      }
	  }
	sortMapByValue(weight_map,tVector);
	int num_quant = std::ceil(0.3 * tVector.size());
	for (int j = 0; j < num_quant; ++j)
	  {
	    int index = tVector[j].first;
	    weight[index] = cur_centroids[i];
	    mask[index] = Dtype(0);
	    modify++;
	  }
	weight_map.clear();
	tVector.clear();
      }
    std::cout << "layer" << param_id << "quantized modify: " << quantized_modify << std::endl;
    std::cout << "layer" << param_id << "threshold modify: " << modify << std::endl;
  }



  

template<typename Dtype>
void SGDSolver<Dtype>::NumberZero(int param_id, const Dtype* weight)
{
  float zero = 0;
  int count_num = 0;
    const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
    int count = net_params[param_id]->count();
    for(int i = 0; i < count; i++)
    {
      if (weight[i] == zero)
	count_num++;
    }
    std::cout<< "layer" << param_id<< "num zero"<< count_num << std::endl;
}
  


template<typename Dtype>
void SGDSolver<Dtype>::PrintValue(int param_id, const Dtype* weight)
{
  std::cout << "layer " << param_id  << std::endl;
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  int count = net_params[param_id]->count();
  //int count = weight->count();
  for(int i = 0; i < count; i++)
    {
      std::cout << weight[i] << std::endl;
    }
}


template<typename Dtype>
void SGDSolver<Dtype>::CheckNumberClusters(int param_id, const Dtype* weight)
{
  float cur_number = 0;
  std::map<float, float> numbers;
  std::map<float, float>::iterator it;
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  int count = net_params[param_id]->count();
  for(int i = 0; i < count; i++)
  {
    cur_number = weight[i];
    it = numbers.find(cur_number);
    if(it == numbers.end())
      numbers.insert (std::pair<float, float>(cur_number, 1));
  }
  std::cout << "layer " << param_id << " number cluster " << numbers.size()<<std::endl; 
}
//quantization for finding centroid
template<typename Dtype>
void SGDSolver<Dtype>::Find_centroid(int param_id, const Dtype* weight, float max_var, int num_cluster, float step_length, Dtype* global_centroids, Dtype* global_centroids_ranges)
{
	float assement_point = max_var / (num_cluster - 1);
	float scan_start = assement_point - assement_point * 0.1;
	float scan_end = assement_point + assement_point * 0.1;
	vector<float> points = Get_assement_points(scan_start, scan_end, assement_point ,step_length);
	float square_error_min = std::numeric_limits<float>::max();
	vector<float> temp_centroids;
	vector<float> temp_centroids_ranges;
	for (int i = 0; i < points.size(); ++i)
	{
		vector<float> centroids = Get_centroids(points[i], num_cluster);
		vector<float> centroids_ranges = Get_centroids_range(centroids);
		float square_error = Calculate_square_error(param_id, centroids, centroids_ranges);
		if (square_error < square_error_min)
		{
			square_error_min = square_error;
			temp_centroids = centroids;
			temp_centroids_ranges = centroids_ranges;
		}
	}
	for (int i = 0; i < temp_centroids.size(); ++i)
	{
	  global_centroids[i] = Dtype(temp_centroids[i]);
	}
	for (int i = 0; i < temp_centroids_ranges.size(); ++i)
	{
	  global_centroids_ranges[i] = Dtype(temp_centroids_ranges[i]);
	}
}

template<typename Dtype>
void SGDSolver<Dtype>::Final_quantization()
{
	// only called in final quantization phase
	if (this->param_.clustering_phase()!="quant")
		return;
	std::cout << "you going to final quantization part"<< std::endl;
	for (int param_id = 0; param_id < this->net_->learnable_params().size(); ++param_id)
	{
		if (!this->net_->has_params_clustering_num()[param_id])
		    	        continue;
		const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
		std::cout << "before final quantization" << std::endl;
		CheckNumberClusters(param_id, net_params[param_id]->cpu_data());
		NumberZero(param_id,  net_params[param_id]->cpu_data());
		
		Dtype * weight = net_params[param_id]->mutable_cpu_data();
		const Dtype * global_centroids = centroids_[param_id]->cpu_data();
		const Dtype * global_centroids_ranges_ = centroids_ranges_[param_id]->cpu_data();
		Project_to_centorid(param_id, weight, global_centroids, global_centroids_ranges_);

		std::cout << "after final quantization" << std::endl;
		CheckNumberClusters(param_id, net_params[param_id]->cpu_data());
		NumberZero(param_id,  net_params[param_id]->cpu_data());
	}

}

template<typename Dtype>
void SGDSolver<Dtype>::Initialization_get_interval()
{

	// only called in admm phase
	if (this->param_.clustering_phase()!="admm")
		return;
	std::cout << "you going to admm quantization initialization part" << std::endl;
    // need to assert that number of clusters is larger than 1
	// code goes here
	// resize number of centroids to number of clusters

	for (int param_id = 0; param_id < this->net_->learnable_params().size();++param_id)
	{
		if (!this->net_->has_params_clustering_num()[param_id])
		    	        continue;
		int num_clusters = this->net_->params_clustering_num()[param_id];
		// in order to process it later.
		const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
		//int count = net_params[param_id]->count();
		const Dtype* weight = net_params[param_id]->cpu_data();
		NumberZero(param_id,  weight);
		CheckNumberClusters(param_id, weight);
		std::cout << "layer"<< param_id << ", weight number: " << net_params[param_id]->count()<<std::endl;
		Dtype *global_centroids = centroids_[param_id]->mutable_cpu_data();
		Dtype *global_centroids_ranges_ = centroids_ranges_[param_id]->mutable_cpu_data();
		float cur_max_var = Find_max(weight, param_id);
		float step_length = 0.001;
		//Find_centroid(param_id, weight, cur_max_var, num_clusters, step_length, global_centroids, global_centroids_ranges_);
		float assement_point = cur_max_var / num_clusters;
         	float scan_start = assement_point - assement_point * 0.1;
	        float scan_end = assement_point + assement_point * 0.1;
	        vector<float> points = Get_assement_points(scan_start, scan_end, assement_point ,step_length);
	        float square_error_min = std::numeric_limits<float>::max();
	        vector<float> temp_centroids;
	        vector<float> temp_centroids_ranges;
	        for (int i = 0; i < points.size(); ++i)
	        {
		    vector<float> centroids = Get_centroids(points[i], num_clusters);
		    vector<float> centroids_ranges = Get_centroids_range(centroids);
		    float square_error = Calculate_square_error(param_id, centroids, centroids_ranges);
		    if (square_error < square_error_min)
		    {
			square_error_min = square_error;
			temp_centroids = centroids;
			temp_centroids_ranges = centroids_ranges;
	            }   
	        }
		std::cout <<param_id << std::endl;
		std::cout <<"centroids" <<temp_centroids.size() << std::endl;
		std::cout <<"ranges"<< temp_centroids_ranges.size() << std::endl;
		std::cout <<"centroids_" <<centroids_[param_id]->count() << std::endl;
		std::cout <<"global_centroids_ranges_"<< centroids_ranges_[param_id]->count() << std::endl;
	        for (int i = 0; i < centroids_[param_id]->count(); ++i)
	        {
		  //if(i < temp_centroids.size())
	              global_centroids[i] = Dtype(temp_centroids[i]);
		   //else
		   //   global_centroids[i] = Dtype(0);
	        }
		for (int i = 0; i < centroids_[param_id]->count(); ++i)
		{
		     std::cout<< global_centroids[i] << std::endl;
		}
	        for (int i = 0; i < centroids_ranges_[param_id]->count(); ++i)
	        {
		  //if(i < temp_centroids_ranges.size())
	              global_centroids_ranges_[i] = Dtype(temp_centroids_ranges[i]);
		  //else
		  //    global_centroids_ranges_[i]= Dtype(0);
		}
		for (int i = 0; i < centroids_ranges_[param_id]->count(); ++i)
		{
		  std::cout<<global_centroids_ranges_[i]<<std::endl;
		}
	}
}


template <typename Dtype>
void SGDSolver<Dtype>::Quantization()
{
	// only called in the end of admm / reaches admm maximal iteration
	if (this->param_.clustering_phase()!="retrain")
		return;
	//int num_layers = this->net_->learnable_params().size();
	//need to recover centroids_, centroids_ranges_

	std::cout << "you going to retrain quantization start part "<< std::endl;
	for (int param_id = 0; param_id < this->net_->learnable_params().size();++param_id)
	{
		if (!this->net_->has_params_clustering_num()[param_id])
		    	        continue;
		const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
		std::cout << "before partial quantization" << std::endl;
		CheckNumberClusters(param_id, net_params[param_id]->cpu_data());
		NumberZero(param_id,  net_params[param_id]->cpu_data());
		Dtype * weight = net_params[param_id]->mutable_cpu_data();
		const Dtype * global_centroids = centroids_[param_id]->cpu_data();
		std::cout << "global centroids" << std::endl;
		//modify here
		const Dtype * global_centroids_ranges = centroids_ranges_[param_id]->cpu_data();
		
		int centroids_count = centroids_[param_id]->count();
		for(int i = 0; i < centroids_count; i++)
		{
		  std::cout << global_centroids[i] << std::endl;
		}
		std::cout << "num zero in mask" << std::endl;
		NumberZero(param_id,  mask_[param_id]->cpu_data());
	        Dtype * mask = mask_[param_id]->mutable_cpu_data();
		std::cout << "start to project threshold" << std::endl;
		//modify here
		Project_threshold(param_id, weight, mask, global_centroids, global_centroids_ranges);
		
		std::cout << "after partial num zero in mask" << std::endl;
		NumberZero(param_id,  mask_[param_id]->cpu_data());
		std::cout << "after partial num zero in weight" << std::endl;
		NumberZero(param_id,  net_params[param_id]->cpu_data());
		//PrintValue(param_id, net_params[param_id]->cpu_data());
		std::cout << "after partial quantization" << std::endl;
		CheckNumberClusters(param_id, net_params[param_id]->cpu_data());
	}
}

template <typename Dtype>
void SGDSolver<Dtype>::KeepPrunedWeightsUnchanged()
{
	if (this->param_.clustering_phase()!="retrain" && this->param_.clustering_phase()!="admm")
	{
		return;
	}
	std::cout << "you going to keep pruned weight unchanged part" << std::endl;
	const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
	switch (Caffe::mode()) {
	  case Caffe::CPU: {
	    // NOT_IMPLEMENTED
	    LOG(FATAL) << "ADMM clustering not supported in CPU mode!";
	    break;
	  }
	  case Caffe::GPU: {
	  #ifndef CPU_ONLY
	      for (int param_id = 0; param_id < this->net_->learnable_params().size();
	           ++param_id) {

	    	  keep_pruned_weights_unchanged_gpu(mask_[param_id]->count(),
	    	                     net_params[param_id]->gpu_data(),
	    	                     mask_[param_id]->mutable_gpu_data());

	      }
	  #else
	      NO_GPU;
	  #endif
	      break;
	    }
	    default:
	      LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
	    }


}

template <typename Dtype>
void SGDSolver<Dtype>::ComputeMask() {
  if (this->param_.pruning_phase() != "retrain")
    return;  // no need to run
  std::cout << "you going to compute mask part" << std::endl;
  const vector<float>& net_params_prune_ratio = this->net_->params_prune_ratio();
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();

  switch (Caffe::mode()) {
  case Caffe::CPU: {
    // NOT_IMPLEMENTED
    LOG(FATAL) << "ADMM pruning not supported in CPU mode!";
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    for (int param_id = 0; param_id < this->net_->learnable_params().size();
         ++param_id) {

      if (!this->net_->has_params_prune_ratio()[param_id])
        continue;

      // get pivot
      wtemp_[param_id]->CopyFrom(*net_params[param_id], false, false);
      // make all values positive
      caffe_gpu_abs(wtemp_[param_id]->count(),
                    wtemp_[param_id]->gpu_data(),
                    wtemp_[param_id]->mutable_gpu_data());

      // find n smallest element
      int n = wtemp_[param_id]->count() * net_params_prune_ratio[param_id];
      Dtype* begin = wtemp_[param_id]->mutable_cpu_data();
      Dtype* end = begin + wtemp_[param_id]->count();
      Dtype* nth = begin + n;
      std::nth_element(begin, nth, end);
      Dtype pivot = wtemp_[param_id]->cpu_data()[n];

      // set mask
      set_mask_gpu(mask_[param_id]->count(),
                   net_params[param_id]->gpu_data(),
                   pivot,
                   mask_[param_id]->mutable_gpu_data());

      LOG(INFO) << "Mask set: param_id " << param_id \
                << " prune ratio " << net_params_prune_ratio[param_id];
    }
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ApplyUpdate() {
  Dtype rate = GetLearningRate();
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << this->iter_
        << ", lr = " << rate;
  }
  ClipGradients();
  for (int param_id = 0; param_id < this->net_->learnable_params().size();
       ++param_id) {
    Normalize(param_id);
    Regularize(param_id);
    ADMM(param_id);
    ApplyMask(param_id);
    ADMMWeightClustering(param_id);
    ComputeUpdateValue(param_id, rate);
  }
  this->net_->Update();
}

template <typename Dtype>
void SGDSolver<Dtype>::Normalize(int param_id) {
  if (this->param_.iter_size() == 1) { return; }
  // Scale gradient to counterbalance accumulation.
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const Dtype accum_normalization = Dtype(1.) / this->param_.iter_size();
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    caffe_scal(net_params[param_id]->count(), accum_normalization,
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    caffe_gpu_scal(net_params[param_id]->count(), accum_normalization,
        net_params[param_id]->mutable_gpu_diff());
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::Regularize(int param_id) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_weight_decay =
      this->net_->params_weight_decay();
  Dtype weight_decay = this->param_.weight_decay();
  string regularization_type = this->param_.regularization_type();
  Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    if (local_decay) {
      if (regularization_type == "L2") {
        // add weight decay
        caffe_axpy(net_params[param_id]->count(),
            local_decay,
            net_params[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff());
      } else if (regularization_type == "L1") {
        caffe_cpu_sign(net_params[param_id]->count(),
            net_params[param_id]->cpu_data(),
            temp_[param_id]->mutable_cpu_data());
        caffe_axpy(net_params[param_id]->count(),
            local_decay,
            temp_[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff());
      } else {
        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
      }
    }
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    if (local_decay) {
      if (regularization_type == "L2") {
        // add weight decay
        caffe_gpu_axpy(net_params[param_id]->count(),
            local_decay,
            net_params[param_id]->gpu_data(),
            net_params[param_id]->mutable_gpu_diff());
      } else if (regularization_type == "L1") {
        caffe_gpu_sign(net_params[param_id]->count(),
            net_params[param_id]->gpu_data(),
            temp_[param_id]->mutable_gpu_data());
        caffe_gpu_axpy(net_params[param_id]->count(),
            local_decay,
            temp_[param_id]->gpu_data(),
            net_params[param_id]->mutable_gpu_diff());
      } else {
        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
      }
    }
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

#ifndef CPU_ONLY
template <typename Dtype>
void abs_min_filter_gpu(int N, Dtype* a, Dtype min);
#endif

#ifndef CPU_ONLY
template <typename Dtype>
void get_centroid_grad_gpu(int N, Dtype* centroid_grads, const Dtype* labels, Dtype* grads);
#endif

#ifndef CPU_ONLY
template <typename Dtype>
void replace_grads_by_centroids_gpu(int N, const Dtype* centroid_grads, const Dtype* labels, Dtype* grads);
#endif



template <typename Dtype>
void SGDSolver<Dtype>::ADMMWeightClustering(int param_id)
{
	if (this->param_.clustering_phase() != "admm" || !this->net_->has_params_clustering_num()[param_id])
	    return;  // no need to run ADMM
	//std::cout<< "you going to admm quantization training part" << std::endl;
	const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
	const vector<float>& net_params_rho = this->net_->params_rho();

	switch (Caffe::mode()) {
	  case Caffe::CPU: {
	    // NOT_IMPLEMENTED
	    LOG(FATAL) << "ADMM pruning not supported in CPU mode!";
	    break;
	  }
	  case Caffe::GPU: {
#ifndef CPU_ONLY
	    // update Z and U every admm_iter iterations
	    if (this->iter_ % this->param_.admm_iter() == 0) {
	      // Z = W + U
	      caffe_gpu_add(zutemp_[param_id]->count(),
	                    net_params[param_id]->gpu_data(), // W
	                    zutemp_[param_id]->gpu_diff(),    // +U
	                    zutemp_[param_id]->mutable_gpu_data());  // Z=

	      // do quantization
	      //wtemp_[param_id]->CopyFrom(*zutemp_[param_id], false, false);
	      const Dtype * global_centroids = centroids_[param_id]->cpu_data();
	      const Dtype * global_centroids_ranges_ = centroids_ranges_[param_id]->cpu_data();
	      std::cout <<"doing project centroids" << std::endl;
	      CheckNumberClusters(param_id, zutemp_[param_id]->cpu_data());
	      Project_to_centorid(param_id, zutemp_[param_id]->mutable_cpu_data(), global_centroids, global_centroids_ranges_);
	      CheckNumberClusters(param_id, zutemp_[param_id]->cpu_data());
	      //PrintValue(param_id, zutemp_[param_id]->cpu_data());
	      const Dtype* weight = net_params[param_id]->cpu_data();
	      NumberZero(param_id,  weight);
	      std::cout << "finish project centroids" << std::endl;
	      // updateCentroid
	      //UpdateCentroids(param_id);
	      // clustering
	      //WeightClustering(param_id,zutemp_[param_id]->mutable_cpu_data());


	      if (this->iter_ != 0) {

	        // update U
	        caffe_gpu_add(zutemp_[param_id]->count(),
	                      zutemp_[param_id]->gpu_diff(),    // U
	                      net_params[param_id]->gpu_data(), // +W
	                      zutemp_[param_id]->mutable_gpu_diff());

	        caffe_gpu_sub(zutemp_[param_id]->count(),
	                      zutemp_[param_id]->gpu_diff(),  // U
	                      zutemp_[param_id]->gpu_data(),  // -Z
	                      zutemp_[param_id]->mutable_gpu_diff());
	      }

	      LOG(INFO) << "ADMM Update Z & U: Iteration " << this->iter_;
	    }

	    // update weights every iteration
	    caffe_gpu_axpy(net_params[param_id]->count(),
	                   Dtype(net_params_rho[param_id]),
	                   net_params[param_id]->gpu_data(),  // W
	                   net_params[param_id]->mutable_gpu_diff());

	    caffe_gpu_axpy(net_params[param_id]->count(),
	                   Dtype(-1.0 * net_params_rho[param_id]),
	                   zutemp_[param_id]->gpu_data(),  // -Z
	                   net_params[param_id]->mutable_gpu_diff());

	    caffe_gpu_axpy(net_params[param_id]->count(),
	                   Dtype(net_params_rho[param_id]),
	                   zutemp_[param_id]->gpu_diff(),  // U
	                   net_params[param_id]->mutable_gpu_diff());
#else
	    NO_GPU;
#endif
	    break;
	  }
	  default:
	    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
	  }

}

template <typename Dtype>
void SGDSolver<Dtype>::ADMM(int param_id) {
  if (this->param_.pruning_phase() != "admm" || !this->net_->has_params_prune_ratio()[param_id])
    return;  // no need to run ADMM
  //std::cout << "you going to ADMM pruning training part" << std::endl;
  const vector<float>& net_params_prune_ratio = this->net_->params_prune_ratio();
  const vector<float>& net_params_rho = this->net_->params_rho();
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();

  switch (Caffe::mode()) {
  case Caffe::CPU: {
    // NOT_IMPLEMENTED
    LOG(FATAL) << "ADMM pruning not supported in CPU mode!";
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    // update Z and U every admm_iter iterations
    if (this->iter_ % this->param_.admm_iter() == 0) {
      // Z = W + U
      caffe_gpu_add(zutemp_[param_id]->count(),
                    net_params[param_id]->gpu_data(), // W
                    zutemp_[param_id]->gpu_diff(),    // +U
                    zutemp_[param_id]->mutable_gpu_data());  // Z=

      // get pivot
      wtemp_[param_id]->CopyFrom(*zutemp_[param_id], false, false);
      // make all values positive
      caffe_gpu_abs(wtemp_[param_id]->count(),
                    wtemp_[param_id]->gpu_data(),
                    wtemp_[param_id]->mutable_gpu_data());

      // find n smallest element
      int n = wtemp_[param_id]->count() * net_params_prune_ratio[param_id];
      Dtype* begin = wtemp_[param_id]->mutable_cpu_data();
      Dtype* end = begin + wtemp_[param_id]->count();
      Dtype* nth = begin + n;
      std::nth_element(begin, nth, end);
      Dtype pivot = wtemp_[param_id]->cpu_data()[n];
      // printf("id:%d, pivot:%f\n", param_id, pivot);

      // filter Z
      abs_min_filter_gpu(zutemp_[param_id]->count(),
                         zutemp_[param_id]->mutable_gpu_data(),
                         pivot);

      if (this->iter_ != 0) {
        LOG(INFO) << "bingo! " << param_id << " " << pivot;
        // update U
        caffe_gpu_add(zutemp_[param_id]->count(),
                      zutemp_[param_id]->gpu_diff(),    // U
                      net_params[param_id]->gpu_data(), // +W
                      zutemp_[param_id]->mutable_gpu_diff());

        caffe_gpu_sub(zutemp_[param_id]->count(),
                      zutemp_[param_id]->gpu_diff(),  // U
                      zutemp_[param_id]->gpu_data(),  // -Z
                      zutemp_[param_id]->mutable_gpu_diff());
      }

      LOG(INFO) << "ADMM Update Z & U: Iteration " << this->iter_;
    }

    // update weights every iteration
    caffe_gpu_axpy(net_params[param_id]->count(),
                   Dtype(net_params_rho[param_id]),
                   net_params[param_id]->gpu_data(),  // W
                   net_params[param_id]->mutable_gpu_diff());

    caffe_gpu_axpy(net_params[param_id]->count(),
                   Dtype(-1.0 * net_params_rho[param_id]),
                   zutemp_[param_id]->gpu_data(),  // -Z
                   net_params[param_id]->mutable_gpu_diff());

    caffe_gpu_axpy(net_params[param_id]->count(),
                   Dtype(net_params_rho[param_id]),
                   zutemp_[param_id]->gpu_diff(),  // U
                   net_params[param_id]->mutable_gpu_diff());
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ApplyMask(int param_id) {
  if ((this->param_.pruning_phase() != "retrain" ||!this->net_->has_params_prune_ratio()[param_id]) && (this->param_.clustering_phase()!="retrain" && this->param_.clustering_phase()!="admm" ))
{
	  // only skip this function if it's neither admm pruning nor clustering
    return;
}
  //std::cout<<"you going to apply mask part" << std::endl;
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();

  switch (Caffe::mode()) {
  case Caffe::CPU: {
    // NOT_IMPLEMENTED
    LOG(FATAL) << "ADMM pruning not supported in CPU mode!";
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
  	if (this->param_.clustering_phase()=="admm")
  	{
  	// mask weights
    caffe_gpu_mul(net_params[param_id]->count(),
                  net_params[param_id]->gpu_data(),
                  mask_[param_id]->gpu_data(),
                  net_params[param_id]->mutable_gpu_data());
    // mask gradients
    caffe_gpu_mul(net_params[param_id]->count(),
                  net_params[param_id]->gpu_diff(),
                  mask_[param_id]->gpu_data(),
                  net_params[param_id]->mutable_gpu_diff());
  	}
	else if (this->param_.clustering_phase()=="retrain")
    {
    	 // mask gradients
    caffe_gpu_mul(net_params[param_id]->count(),
                  net_params[param_id]->gpu_diff(),
                  mask_[param_id]->gpu_data(),
                  net_params[param_id]->mutable_gpu_diff());
    }
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

#ifndef CPU_ONLY
template <typename Dtype>
void sgd_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate);
#endif

template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype momentum = this->param_.momentum();
  Dtype local_rate = rate * net_params_lr[param_id];
  // Compute the update to history, then copy it to the parameter diff.
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->cpu_diff(), momentum,
              history_[param_id]->mutable_cpu_data());
    caffe_copy(net_params[param_id]->count(),
        history_[param_id]->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    sgd_update_gpu(net_params[param_id]->count(),
        net_params[param_id]->mutable_gpu_diff(),
        history_[param_id]->mutable_gpu_data(),
        momentum, local_rate);
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverState(const string& model_filename) {
  switch (this->param_.snapshot_format()) {
    case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
      SnapshotSolverStateToBinaryProto(model_filename);
      break;
    case caffe::SolverParameter_SnapshotFormat_HDF5:
      SnapshotSolverStateToHDF5(model_filename);
      break;
    default:
      LOG(FATAL) << "Unsupported snapshot format.";
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverStateToBinaryProto(
    const string& model_filename) {
  SolverState state;
  state.set_iter(this->iter_);
  state.set_learned_net(model_filename);
  state.set_current_step(this->current_step_);
  state.clear_history();
  for (int i = 0; i < history_.size(); ++i) {
    // Add history
    BlobProto* history_blob = state.add_history();
    history_[i]->ToProto(history_blob);
  }
  string snapshot_filename = Solver<Dtype>::SnapshotFilename(".solverstate");
  LOG(INFO)
    << "Snapshotting solver state to binary proto file " << snapshot_filename;
  WriteProtoToBinaryFile(state, snapshot_filename.c_str());

  // snapshot admm state
  if (this->param_.pruning_phase() == "admm")
    SnapshotADMMStateToBinaryProto();

  if (this->param_.clustering_phase() == "admm") {
    SnapshotADMMStateToBinaryProto();
    SnapshotCentroidsStateToBinaryProto();
    SnapshotCentroidsRangeStateToBinaryProto();
  }
}

//for quantization snapshot for centroids
template <typename Dtype>
void SGDSolver<Dtype>::SnapshotCentroidsStateToBinaryProto()
{
  CentroidsState state;
  state.clear_centroids();
  for (int i = 0; i < centroids_.size(); ++i) {
    //if (!this->net_->has_params_clustering_num()[i])
      //  continue;
    BlobProto* centroids_blob = state.add_centroids();
    centroids_[i]->ToProto(centroids_blob, true);
  }
  string snapshot_filename = Solver<Dtype>::SnapshotFilename(".solverstate.centroids");
  LOG(INFO)
    << "Snapshotting centroids state to binary proto file " << snapshot_filename;
  WriteProtoToBinaryFile(state, snapshot_filename.c_str());
}

//for quantization snapshot for centroids
template <typename Dtype>
void SGDSolver<Dtype>::RestoreCentroidsStateFromBinaryProto(const string& state_file)
{
  CentroidsState state;
  ReadProtoFromBinaryFile(state_file, &state);
  CHECK_EQ(state.centroids_size(), centroids_.size())
      << "Incorrect length of centroids blobs.";
  LOG(INFO) << "SGDSolver: restoring centroids";
  for (int i = 0; i <  centroids_.size(); ++i) {
    //if (!this->net_->has_params_clustering_num()[i])
    //  continue;
    centroids_[i]->FromProto(state.centroids(i));
  }
}
//for quantization snapshot for centroids ranges
template <typename Dtype>
void SGDSolver<Dtype>::SnapshotCentroidsRangeStateToBinaryProto()
{
  CentroidsRangeState state;
  state.clear_range();
  for (int i = 0; i < centroids_ranges_.size(); ++i) {
    //if (!this->net_->has_params_clustering_num()[i])
    //  continue;
    BlobProto* centroidsRange_blob = state.add_range();
    centroids_ranges_[i]->ToProto(centroidsRange_blob, true);
  }
  string snapshot_filename = Solver<Dtype>::SnapshotFilename(".solverstate.centroidsrange");
  LOG(INFO)
    << "Snapshotting centroids range state to binary proto file " << snapshot_filename;
  WriteProtoToBinaryFile(state, snapshot_filename.c_str());
}

//for quantization snapshot for centroids ranges
template <typename Dtype>
void SGDSolver<Dtype>::RestoreCentroidsRangeStateFromBinaryProto(const string& state_file)
{
  CentroidsRangeState state;
  ReadProtoFromBinaryFile(state_file, &state);
  CHECK_EQ(state.range_size(), centroids_ranges_.size())
      << "Incorrect length of centroids_ranges_ blobs.";
  LOG(INFO) << "SGDSolver: restoring centroids_ranges_";
  for (int i = 0; i < centroids_ranges_.size(); ++i) {
    //if (!this->net_->has_params_clustering_num()[i])
    //  continue;
    centroids_ranges_[i]->FromProto(state.range(i));
  }
}


template <typename Dtype>
void SGDSolver<Dtype>::SnapshotADMMStateToBinaryProto() {
  ADMMState state;
  state.clear_zutemp();
  for (int i = 0; i < zutemp_.size(); ++i) {
    BlobProto* zutemp_blob = state.add_zutemp();
    zutemp_[i]->ToProto(zutemp_blob, true);
  }
  string snapshot_filename = Solver<Dtype>::SnapshotFilename(".solverstate.admm");
  LOG(INFO)
    << "Snapshotting admm state to binary proto file " << snapshot_filename;
  WriteProtoToBinaryFile(state, snapshot_filename.c_str());
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverStateToHDF5(
    const string& model_filename) {
  string snapshot_filename =
      Solver<Dtype>::SnapshotFilename(".solverstate.h5");
  LOG(INFO) << "Snapshotting solver state to HDF5 file " << snapshot_filename;
  hid_t file_hid = H5Fcreate(snapshot_filename.c_str(), H5F_ACC_TRUNC,
      H5P_DEFAULT, H5P_DEFAULT);
  CHECK_GE(file_hid, 0)
      << "Couldn't open " << snapshot_filename << " to save solver state.";
  hdf5_save_int(file_hid, "iter", this->iter_);
  hdf5_save_string(file_hid, "learned_net", model_filename);
  hdf5_save_int(file_hid, "current_step", this->current_step_);
  hid_t history_hid = H5Gcreate2(file_hid, "history", H5P_DEFAULT, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(history_hid, 0)
      << "Error saving solver state to " << snapshot_filename << ".";
  for (int i = 0; i < history_.size(); ++i) {
    ostringstream oss;
    oss << i;
    hdf5_save_nd_dataset<Dtype>(history_hid, oss.str(), *history_[i]);
  }
  H5Gclose(history_hid);
  H5Fclose(file_hid);
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverStateFromBinaryProto(
    const string& state_file) {
  SolverState state;
  ReadProtoFromBinaryFile(state_file, &state);
  this->iter_ = state.iter();
  if (state.has_learned_net()) {
    NetParameter net_param;
    ReadNetParamsFromBinaryFileOrDie(state.learned_net().c_str(), &net_param);
    this->net_->CopyTrainedLayersFrom(net_param);
  }
  this->current_step_ = state.current_step();
  CHECK_EQ(state.history_size(), history_.size())
      << "Incorrect length of history blobs.";
  LOG(INFO) << "SGDSolver: restoring history";
  for (int i = 0; i < history_.size(); ++i) {
    history_[i]->FromProto(state.history(i));
  }

  // snapshot admm state
  if (this->param_.pruning_phase() == "admm")
    RestoreADMMStateFromBinaryProto(state_file + ".admm");

  if (this->param_.clustering_phase() == "admm") {
    RestoreADMMStateFromBinaryProto(state_file + ".admm");
  }
  if (this->param_.clustering_phase() == "retrain") {
    RestoreCentroidsStateFromBinaryProto(state_file + ".centroids");
    RestoreCentroidsRangeStateFromBinaryProto(state_file + ".centroidsrange");
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreADMMStateFromBinaryProto(const string& state_file) {
  ADMMState state;
  ReadProtoFromBinaryFile(state_file, &state);
  CHECK_EQ(state.zutemp_size(), zutemp_.size())
      << "Incorrect length of zutemp blobs.";
  LOG(INFO) << "SGDSolver: restoring zutemp";
  for (int i = 0; i < zutemp_.size(); ++i) {
    zutemp_[i]->FromProto(state.zutemp(i));
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverStateFromHDF5(const string& state_file) {
  hid_t file_hid = H5Fopen(state_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open solver state file " << state_file;
  this->iter_ = hdf5_load_int(file_hid, "iter");
  if (H5LTfind_dataset(file_hid, "learned_net")) {
    string learned_net = hdf5_load_string(file_hid, "learned_net");
    this->net_->CopyTrainedLayersFrom(learned_net);
  }
  this->current_step_ = hdf5_load_int(file_hid, "current_step");
  hid_t history_hid = H5Gopen2(file_hid, "history", H5P_DEFAULT);
  CHECK_GE(history_hid, 0) << "Error reading history from " << state_file;
  int state_history_size = hdf5_get_num_links(history_hid);
  CHECK_EQ(state_history_size, history_.size())
      << "Incorrect length of history blobs.";
  for (int i = 0; i < history_.size(); ++i) {
    ostringstream oss;
    oss << i;
    hdf5_load_nd_dataset<Dtype>(history_hid, oss.str().c_str(), 0,
                                kMaxBlobAxes, history_[i].get());
  }
  H5Gclose(history_hid);
  H5Fclose(file_hid);
}


INSTANTIATE_CLASS(SGDSolver);
REGISTER_SOLVER_CLASS(SGD);

}  // namespace caffe
