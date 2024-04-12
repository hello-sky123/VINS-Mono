#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>
#include <utility>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
  ResidualBlockInfo(ceres::CostFunction* _cost_function, ceres::LossFunction* _loss_function,
                    std::vector<double*> _parameter_blocks, std::vector<int> _drop_set)
  : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(std::move(_parameter_blocks)), drop_set(std::move(_drop_set)) {}

  void Evaluate();

  ceres::CostFunction* cost_function;
  ceres::LossFunction* loss_function;
  std::vector<double*> parameter_blocks;
  std::vector<int> drop_set;

  double** raw_jacobians;
  std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
  Eigen::VectorXd residuals;

  int localSize(int size)
  {
    return size == 7 ? 6 : size;
  }
};

struct ThreadsStruct
{
  std::vector<ResidualBlockInfo*> sub_factors;
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  std::unordered_map<long, int> parameter_block_size; //global size
  std::unordered_map<long, int> parameter_block_idx; //local size
};

class MarginalizationInfo
{
 public:
   ~MarginalizationInfo();
   static int localSize(int size) ;
   int globalSize(int size) const;
   void addResidualBlockInfo(ResidualBlockInfo* residual_block_info);
   void preMarginalize();
   void marginalize();
   std::vector<double*> getParameterBlocks(std::unordered_map<long, double*>& addr_shift);

   std::vector<ResidualBlockInfo*> factors;
   int m, n;
   std::unordered_map<long, int> parameter_block_size; //global size // 地址->global size
   int sum_block_size;
   std::unordered_map<long, int> parameter_block_idx; //local size // 地址->参数排列的顺序idx
   std::unordered_map<long, double*> parameter_block_data; // 地址->参数块实际内容的地址

   std::vector<int> keep_block_size; //global size
   std::vector<int> keep_block_idx;  //local size
   std::vector<double*> keep_block_data;

   Eigen::MatrixXd linearized_jacobians;
   Eigen::VectorXd linearized_residuals;
   const double eps = 1e-8;

};

class MarginalizationFactor : public ceres::CostFunction // 由于边缘化的costfuntion不是固定大小的，因此只能继承最基本的类
{
 public:
   explicit MarginalizationFactor(MarginalizationInfo* _marginalization_info);
   bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

   MarginalizationInfo* marginalization_info;
};
