#include <camodocal/sparse_graph/Transform.h>

namespace camodocal
{

Transform::Transform() // 默认构造函数
{
  m_q.setIdentity(); // 设置为单位四元数
  m_t.setZero(); // 设置为零向量
}

Transform::Transform(const Eigen::Matrix4d& H)
{
  m_q = Eigen::Quaterniond(H.block<3, 3>(0,0)); // 从H中提取旋转矩阵
  m_t = H.block<3,1>(0,3);
}

Eigen::Quaterniond& Transform::rotation()
{
  return m_q;
}

const Eigen::Quaterniond& Transform::rotation() const
{
  return m_q;
}

double* Transform::rotationData() // 返回四元数的数据指针
{
    return m_q.coeffs().data();
}

const double* Transform::rotationData() const
{
  return m_q.coeffs().data();
}

Eigen::Vector3d& Transform::translation() // 返回平移向量
{
  return m_t;
}

const Eigen::Vector3d& Transform::translation() const
{
  return m_t;
}

double* Transform::translationData()
{
  return m_t.data();
}

const double* Transform::translationData() const
{
  return m_t.data();
}

Eigen::Matrix4d Transform::toMatrix() const
{
  Eigen::Matrix4d H;
  H.setIdentity();
  H.block<3,3>(0,0) = m_q.toRotationMatrix();
  H.block<3,1>(0,3) = m_t;

  return H;
}

}
