#pragma once

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../parameters.h"

class ProjectionTdFactor: public ceres::SizedCostFunction<2, 7, 7, 7, 1, 1> // 2是残差维度，7、7、7是两个位姿和一个外参，1是逆深度，1是时间差td
{
 public:
   ProjectionTdFactor(Eigen::Vector3d  _pts_i, Eigen::Vector3d  _pts_j,
                      const Eigen::Vector2d& _velocity_i, const Eigen::Vector2d& _velocity_j,
                      double _td_i, double _td_j, double _row_i, double _row_j);
   bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override;
   void check(double** parameters);

   Eigen::Vector3d pts_i, pts_j;
   Eigen::Vector3d velocity_i, velocity_j;
   double td_i, td_j;
   Eigen::Matrix<double, 2, 3> tangent_base;
   double row_i, row_j;
   static Eigen::Matrix2d sqrt_info;
   static double sum_t;
};
