#include "utility.h"

Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d& g)
{
  Eigen::Matrix3d R0;
  Eigen::Vector3d ng1 = g.normalized(); // 重力向量的单位向量
  Eigen::Vector3d ng2{0, 0, 1.0};
  R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix(); // R0 * ng1 = ng2
  double yaw = Utility::R2ypr(R0).x(); // 旋转矩阵转欧拉角，顺序是yaw pitch roll
  R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0; // yaw角清0，pitch和roll不变
  // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
  return R0;
}
