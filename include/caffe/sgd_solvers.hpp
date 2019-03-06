#ifndef CAFFE_SGD_SOLVERS_HPP_
#define CAFFE_SGD_SOLVERS_HPP_

#include <string>
#include <vector>
//modify here
#include <map>

#include "caffe/solver.hpp"

namespace caffe {

/**
 * @brief Optimizes the parameters of a Net using
 *        stochastic gradient descent (SGD) with momentum.
 */
template <typename Dtype>
class SGDSolver : public Solver<Dtype> {
 public:
  explicit SGDSolver(const SolverParameter& param)
      : Solver<Dtype>(param) { PreSolve(); }
  explicit SGDSolver(const string& param_file)
      : Solver<Dtype>(param_file) { PreSolve(); }
  virtual inline const char* type() const { return "SGD"; }

  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }

 protected:
  void PreSolve();
  Dtype GetLearningRate();
  //modify here
  virtual void sortMapByValue(map<int, float>& tMap,vector<std::pair<int, float> >& tVector);
  virtual bool is_weight_quant_zero(int centroid_index, float cur_centroids, float weight, const Dtype * cur_centroids_ranges, int centroids_ranges_count);
  
  virtual void ApplyUpdate();
  virtual void Normalize(int param_id);
  virtual void Regularize(int param_id);
  // ADMM
  virtual void ADMM(int param_id);
  virtual void ComputeMask();
  virtual void KeepPrunedWeightsUnchanged();
  virtual void Final_quantization();
  virtual void ApplyMask(int param_id);
  virtual void SnapshotADMMStateToBinaryProto();
  virtual void RestoreADMMStateFromBinaryProto(const string& state_file);
  virtual void ADMMWeightClustering(int param_id);
  //virtual void CentroidRetrain(int param_id);
  // update centroids
  //virtual void UpdateCentroids(get_centroids_range(centroids));
  // Initialization_get_interval update labelMatrix
  virtual void Initialization_get_interval();
  //virtual void WeightClustering(int param_id,Dtype* weight);
  
  // project to centroid
  virtual void Project_to_centorid(int param_id, Dtype* weight, const Dtype* global_centroids, const Dtype* c_range);
   // partial project to centroid
  //modify here
  virtual void Project_threshold(int param_id,Dtype* weight, Dtype* mask, const Dtype* global_centroids, const Dtype * global_centroids_ranges);

  // call weight clustering in every layer that defines clustering
  virtual void Quantization();
  
  //weight quantization find max weight
  virtual float Find_max(const Dtype* weight, int param_id);
  virtual void Find_centroid(int param_id, const Dtype* weight, float max_var, int num_cluster, float step_length, Dtype* global_centroids, Dtype* global_centroids_ranges);
  virtual vector<float> Get_assement_points(float s_start, float s_end, float assement_point, float step);
  virtual vector<float> Get_centroids(float scale, int one_side_clusters);
  virtual vector<float> Get_centroids_range(vector<float>& centroids);
  virtual float Calculate_square_error(int param_id,  vector<float>& cur_centroids, vector<float>& c_range);
  virtual void SnapshotCentroidsStateToBinaryProto();
  virtual void RestoreCentroidsStateFromBinaryProto(const string& state_file);
  virtual void SnapshotCentroidsRangeStateToBinaryProto();
  virtual void RestoreCentroidsRangeStateFromBinaryProto(const string& state_file);
  virtual void CheckNumberClusters(int param_id, const Dtype* weight);
  virtual void PrintValue(int param_id, const Dtype* weight);
  virtual void NumberZero(int param_id, const Dtype* weight);

  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  virtual void ClipGradients();
  virtual void SnapshotSolverState(const string& model_filename);
  virtual void SnapshotSolverStateToBinaryProto(const string& model_filename);
  virtual void SnapshotSolverStateToHDF5(const string& model_filename);
  virtual void RestoreSolverStateFromHDF5(const string& state_file);
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file);
  // save and restore labels
  //virtual void SnapshotLabelsToBinaryProto();
  //virtual void RestoreLabelsFromBinaryProto(const string& state_file);
  // history maintains the historical momentum data.
  // update maintains update related data and is not needed in snapshots.
  // temp maintains other information that might be needed in computation
  //   of gradients/updates and is not needed in snapshots
  vector<shared_ptr<Blob<Dtype> > > history_, update_, temp_;
  vector<shared_ptr<Blob<Dtype> > > wtemp_, zutemp_; // Z in blob data, U in blob diff
  vector<shared_ptr<Blob<Dtype> > > mask_;
  //vector<shared_ptr<Blob<Dtype> > > labelMatrix_;
  //vector<shared_ptr<Blob<Dtype> > > centroid_grads_;
  vector<shared_ptr<Blob<Dtype> > > centroids_;
  vector<shared_ptr<Blob<Dtype> > > centroids_ranges_;
  
  //vector<shared_ptr<Blob<Dtype> > > centroid_population_;
  DISABLE_COPY_AND_ASSIGN(SGDSolver);
};

template <typename Dtype>
class NesterovSolver : public SGDSolver<Dtype> {
 public:
  explicit NesterovSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) {}
  explicit NesterovSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) {}
  virtual inline const char* type() const { return "Nesterov"; }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(NesterovSolver);
};

template <typename Dtype>
class AdaGradSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaGradSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { constructor_sanity_check(); }
  explicit AdaGradSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { constructor_sanity_check(); }
  virtual inline const char* type() const { return "AdaGrad"; }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with AdaGrad.";
  }

  DISABLE_COPY_AND_ASSIGN(AdaGradSolver);
};


template <typename Dtype>
class RMSPropSolver : public SGDSolver<Dtype> {
 public:
  explicit RMSPropSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { constructor_sanity_check(); }
  explicit RMSPropSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { constructor_sanity_check(); }
  virtual inline const char* type() const { return "RMSProp"; }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with RMSProp.";
    CHECK_GE(this->param_.rms_decay(), 0)
        << "rms_decay should lie between 0 and 1.";
    CHECK_LT(this->param_.rms_decay(), 1)
        << "rms_decay should lie between 0 and 1.";
  }

  DISABLE_COPY_AND_ASSIGN(RMSPropSolver);
};

template <typename Dtype>
class AdaDeltaSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaDeltaSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { AdaDeltaPreSolve(); }
  explicit AdaDeltaSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { AdaDeltaPreSolve(); }
  virtual inline const char* type() const { return "AdaDelta"; }

 protected:
  void AdaDeltaPreSolve();
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(AdaDeltaSolver);
};

/**
 * @brief AdamSolver, an algorithm for first-order gradient-based optimization
 *        of stochastic objective functions, based on adaptive estimates of
 *        lower-order moments. Described in [1].
 *
 * [1] D. P. Kingma and J. L. Ba, "ADAM: A Method for Stochastic Optimization."
 *     arXiv preprint arXiv:1412.6980v8 (2014).
 */
template <typename Dtype>
class AdamSolver : public SGDSolver<Dtype> {
 public:
  explicit AdamSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { AdamPreSolve();}
  explicit AdamSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { AdamPreSolve(); }
  virtual inline const char* type() const { return "Adam"; }

 protected:
  void AdamPreSolve();
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(AdamSolver);
};

}  // namespace caffe

#endif  // CAFFE_SGD_SOLVERS_HPP_
