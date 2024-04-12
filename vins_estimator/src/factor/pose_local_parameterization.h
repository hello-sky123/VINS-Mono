#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "../utility/utility.h"

// 自定义的更新方式需要继承ceres::LocalParameterization
class PoseLocalParameterization : public ceres::LocalParameterization
{
  bool Plus(const double* x, const double* delta, double* x_plus_delta) const override;
  bool ComputeJacobian(const double* x, double* jacobian) const override;
  int GlobalSize() const override { return 7; }; // Size of x 旋转的参数化方式是四元数，所以是7
  int LocalSize() const override { return 6; }; // Size of delta 旋转局部参数化是旋转向量，所以是6
};
