#ifndef EIGENQUATERNIONPARAMETERIZATION_H
#define EIGENQUATERNIONPARAMETERIZATION_H

#include "ceres/local_parameterization.h"

namespace camodocal
{
// 用于处理四元数的局部参数化，来保持模始终为1
class EigenQuaternionParameterization : public ceres::LocalParameterization
{
 public:
   ~EigenQuaternionParameterization() override = default;
   bool Plus(const double* x, const double* delta, double* x_plus_delta) const override;
   bool ComputeJacobian(const double* x, double* jacobian) const override;
   int GlobalSize() const override { return 4; } // GlobalSize()返回优化变量x的维度
   int LocalSize() const override { return 3; } // LocalSize()返回增量delta的维度

 private:
   template<typename T>
   void EigenQuaternionProduct(const T z[4], const T w[4], T zw[4]) const;
};


template<typename T> // Eigen的四元数顺序为x,y,z,w
void EigenQuaternionParameterization::EigenQuaternionProduct(const T z[4], const T w[4], T zw[4]) const
{
  zw[0] = z[3] * w[0] + z[0] * w[3] + z[1] * w[2] - z[2] * w[1];
  zw[1] = z[3] * w[1] - z[0] * w[2] + z[1] * w[3] + z[2] * w[0];
  zw[2] = z[3] * w[2] + z[0] * w[1] - z[1] * w[0] + z[2] * w[3];
  zw[3] = z[3] * w[3] - z[0] * w[0] - z[1] * w[1] - z[2] * w[2];
}

}

#endif

