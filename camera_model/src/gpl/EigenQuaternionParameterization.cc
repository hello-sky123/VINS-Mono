#include "camodocal/gpl/EigenQuaternionParameterization.h"

#include <cmath>

namespace camodocal
{

bool EigenQuaternionParameterization::Plus(const double* x, const double* delta, double* x_plus_delta) const
{
  const double norm_delta = sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]); // 旋转向量标示的四元数的更新量
  if (norm_delta > 0.0) // norm_delta表示旋转的角度
  {
    const double sin_delta_by_delta = (sin(norm_delta / 2.0) / norm_delta);
    double q_delta[4];
    q_delta[0] = sin_delta_by_delta * delta[0];
    q_delta[1] = sin_delta_by_delta * delta[1];
    q_delta[2] = sin_delta_by_delta * delta[2];
    q_delta[3] = cos(norm_delta / 2.0);
    EigenQuaternionProduct(q_delta, x, x_plus_delta);
  }
  else
  {
    for (int i = 0; i < 4; ++i)
    {
      x_plus_delta[i] = x[i];
    }
  }
  return true;
}

// 这个雅可比是更新后的四元数对更新量求偏导数(维度为4*3)
bool EigenQuaternionParameterization::ComputeJacobian(const double* x, double* jacobian) const
{
  jacobian[0] =  x[3]; jacobian[1]  =  x[2]; jacobian[2]  = -x[1];  // NOLINT忽略本行代码的检查
  jacobian[3] = -x[2]; jacobian[4]  =  x[3]; jacobian[5]  =  x[0];  // NOLINT
  jacobian[6] =  x[1]; jacobian[7] = -x[0]; jacobian[8] =  x[3];  // NOLINT
  jacobian[9] = -x[0]; jacobian[10]  = -x[1]; jacobian[11]  = -x[2];  // NOLINT
  return true;
}

}
