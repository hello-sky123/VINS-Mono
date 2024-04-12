#include "initial_ex_rotation.h"

InitialEXRotation::InitialEXRotation(){
  frame_count = 0;
  Rc.emplace_back(Matrix3d::Identity());
  Rc_g.emplace_back(Matrix3d::Identity());
  Rimu.emplace_back(Matrix3d::Identity());
  ric = Matrix3d::Identity();
}

bool InitialEXRotation::CalibrationExRotation(const vector<pair<Vector3d, Vector3d>>& corres, const Quaterniond& delta_q_imu, Matrix3d& calib_ric_result)
{
  frame_count++;
  // 根据特征点关联信息，求解两个连续帧之间的相对旋转
  Rc.push_back(solveRelativeR(corres));
  Rimu.push_back(delta_q_imu.toRotationMatrix());
  // 通过外参把imu坐标系下的旋转转换到相机坐标系下
  Rc_g.emplace_back(ric.inverse() * delta_q_imu * ric); // ric是上一次求解得到的外参

  Eigen::MatrixXd A(frame_count * 4, 4); // 4N x 4的矩阵
  A.setZero();
  int sum_ok = 0;
  for (int i = 1; i <= frame_count; i++)
  {
    Quaterniond r1(Rc[i]); // 由图像信息得到的相对于第一个图像帧的第k帧相机坐标系的位姿
    Quaterniond r2(Rc_g[i]); // 由imu得到的相对于第一个图像帧的第k帧相机坐标系的位姿

    double angular_distance = 180 / M_PI * r1.angularDistance(r2); // 两个旋转之间的角度差（rad）
    ROS_DEBUG("%d %f", i, angular_distance);

    double huber = angular_distance > 5.0 ? 5.0 / angular_distance: 1.0;
    ++sum_ok;
    Matrix4d L, R;

    double w = Quaterniond(Rc[i]).w();
    Vector3d q = Quaterniond(Rc[i]).vec();
    L.block<3, 3>(0, 0) = w * Matrix3d::Identity() + Utility::skewSymmetric(q);
    L.block<3, 1>(0, 3) = q;
    L.block<1, 3>(3, 0) = -q.transpose();
    L(3, 3) = w;

    Quaterniond R_ij(Rimu[i]);
    w = R_ij.w();
    q = R_ij.vec();
    R.block<3, 3>(0, 0) = w * Matrix3d::Identity() - Utility::skewSymmetric(q);
    R.block<3, 1>(0, 3) = q;
    R.block<1, 3>(3, 0) = -q.transpose();
    R(3, 3) = w;

    A.block<4, 4>((i - 1) * 4, 0) = huber * (L - R);
  }

  JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
  Matrix<double, 4, 1> x = svd.matrixV().col(3);
  Quaterniond estimated_R(x);
  ric = estimated_R.toRotationMatrix().inverse();
  //cout << svd.singularValues().transpose() << endl;
  //cout << ric << endl;
  Vector3d ric_cov;
  ric_cov = svd.singularValues().tail<3>(); // 取后三个奇异值
  // 因为旋转是3个自由度，因此检查第三个奇异值是否大于0.25（通常需要足够的运动激励才能保证得到没有奇异的解），同时检查是否已经达到了滑窗大小
  if (frame_count >= WINDOW_SIZE && ric_cov(1) > 0.25)
  {
    calib_ric_result = ric;
    return true;
  }
  else
    return false;
}

Matrix3d InitialEXRotation::solveRelativeR(const vector<pair<Vector3d, Vector3d>>& corres)
{
  if (corres.size() >= 9)
  {
    vector<cv::Point2f> ll, rr;
    for (const auto& corre: corres)
    {
        ll.emplace_back(corre.first(0), corre.first(1));
        rr.emplace_back(corre.second(0), corre.second(1));
    }
    cv::Mat E = cv::findFundamentalMat(ll, rr);
    cv::Mat_<double> R1, R2, t1, t2; // 它是cv::Mat的一个封装。它允许在编译时确定矩阵的数据类型
    decomposeE(E, R1, R2, t1, t2);

    if (determinant(R1) + 1.0 < 1e-09)
    {
      E = -E;
      decomposeE(E, R1, R2, t1, t2);
    }
    // 比较正深度的比例
    double ratio1 = max(testTriangulation(ll, rr, R1, t1), testTriangulation(ll, rr, R1, t2));
    double ratio2 = max(testTriangulation(ll, rr, R2, t1), testTriangulation(ll, rr, R2, t2));
    cv::Mat_<double> ans_R_cv = ratio1 > ratio2 ? R1: R2;

    // 这里解出来的是R21
    Matrix3d ans_R_eigen;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        ans_R_eigen(j, i) = ans_R_cv(i, j); // 这里转换成R12
    return ans_R_eigen;
  }
  return Matrix3d::Identity();
}

/**
 * @brief 通过三角化检验R和t的正确性
 * @param[in] l l相机的观测
 * @param[in] r r相机的观测
 * @param[in] R 旋转矩阵
 * @param[in] t 平移向量
 * @return double
 */
double InitialEXRotation::testTriangulation(const vector<cv::Point2f>& l,const vector<cv::Point2f>& r,
                                            cv::Mat_<double> R, cv::Mat_<double> t)
{
  cv::Mat pointcloud;
  // 其中一帧设置为单位阵
  cv::Matx34d P = cv::Matx34d(1, 0, 0, 0,
                              0, 1, 0, 0,
                              0, 0, 1, 0);
  cv::Matx34d P1 = cv::Matx34d(R(0, 0), R(0, 1), R(0, 2), t(0),
                               R(1, 0), R(1, 1), R(1, 2), t(1),
                               R(2, 0), R(2, 1), R(2, 2), t(2));
  //
  cv::triangulatePoints(P, P1, l, r, pointcloud);
  int front_count = 0;
  for (int i = 0; i < pointcloud.cols; i++) // 得到的三角化的特征点是4xN的矩阵，每一列是一个特征点的齐次坐标
  {
    double normal_factor = pointcloud.col(i).at<float>(3);

    cv::Mat_<double> p_3d_l = cv::Mat(P) * (pointcloud.col(i) / normal_factor);
    cv::Mat_<double> p_3d_r = cv::Mat(P1) * (pointcloud.col(i) / normal_factor);
    if (p_3d_l(2) > 0 && p_3d_r(2) > 0)
      front_count++;
  }
  ROS_DEBUG("MotionEstimator: %f", 1.0 * front_count / pointcloud.cols);
  return 1.0 * front_count / pointcloud.cols;
}

void InitialEXRotation::decomposeE(const cv::Mat& E, cv::Mat_<double>& R1, cv::Mat_<double>& R2,
                                   cv::Mat_<double>& t1, cv::Mat_<double>& t2)
{
  cv::SVD svd(E, cv::SVD::MODIFY_A);
  cv::Matx33d W(0, -1, 0,
                1, 0, 0,
                0, 0, 1);
  cv::Matx33d Wt(0, 1, 0,
                 -1, 0, 0,
                 0, 0, 1);
  R1 = svd.u * cv::Mat(W) * svd.vt;
  R2 = svd.u * cv::Mat(Wt) * svd.vt;
  t1 = svd.u.col(2);
  t2 = -svd.u.col(2);
}
