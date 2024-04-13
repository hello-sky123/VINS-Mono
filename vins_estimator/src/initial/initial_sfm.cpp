#include "initial_sfm.h"

GlobalSFM::GlobalSFM(){}

/**
 * @brief 三角化特征点
 * @param Pose0
 * @param Pose1
 * @param point0
 * @param point1
 * @param point_3d
 * 约束方程为：x0 x (P0 * X) = 0, x1 x (P1 * X) = 0
 */
void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4>& Pose0, Eigen::Matrix<double, 3, 4>& Pose1,
						                     Vector2d& point0, Vector2d& point1, Vector3d& point_3d)
{
  Matrix4d design_matrix = Matrix4d::Zero();
  design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
  design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
  design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
  design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);

  // 使用jacobiSvd求解Ax=0的最小二乘解，用雅可比迭代进行奇异值分解
  Vector4d triangulated_point;
  triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>(); // rightCols<1>()表示取最后一列
  point_3d(0) = triangulated_point(0) / triangulated_point(3);
  point_3d(1) = triangulated_point(1) / triangulated_point(3);
  point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

// 根据已经三角化的空间点以及其在当前帧的观测，求解当前帧的位姿
bool GlobalSFM::solveFrameByPnP(Matrix3d& R_initial, Vector3d& P_initial, int i, vector<SFMFeature>& sfm_f) const
{
  vector<cv::Point2f> pts_2_vector;
  vector<cv::Point3f> pts_3_vector;
  for (int j = 0; j < feature_num; j++)
  {
    if (!sfm_f[j].state)
      continue;
    
    for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
    {
      if (sfm_f[j].observation[k].first == i) // 该特征点在第i帧中有观测
      {
        Vector2d img_pts = sfm_f[j].observation[k].second;
        cv::Point2f pts_2((float)img_pts(0), (float)img_pts(1));
        pts_2_vector.push_back(pts_2);
        cv::Point3f pts_3((float)sfm_f[j].position[0], (float)sfm_f[j].position[1], (float)sfm_f[j].position[2]);
        pts_3_vector.push_back(pts_3);
        break;
      }
    }
  }

  if (int(pts_2_vector.size()) < 15)
  {
    printf("unstable features tracking, please slowly move you device!\n");
    if (int(pts_2_vector.size()) < 10)
      return false;
  }

  cv::Mat r, rvec, t, D, tmp_r;
  cv::eigen2cv(R_initial, tmp_r);
  cv::Rodrigues(tmp_r, rvec);
  cv::eigen2cv(P_initial, t);
  // 使用归一化的2d时，相机内参矩阵K为单位阵
  cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
  
  bool pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, true); // 使用迭代优化的方式求解PnP问题，得到Tcw
  if(!pnp_succ)
  {
    return false;
  }

  cv::Rodrigues(rvec, r);
  MatrixXd R_pnp;
  cv::cv2eigen(r, R_pnp);
  MatrixXd T_pnp;
  cv::cv2eigen(t, T_pnp);
  R_initial = R_pnp;
  P_initial = T_pnp;
  return true;

}

/**
 * @brief 根据两帧索引和位姿三角化两帧之间的特征点
 * @param[in] frame0
 * @param[in] Pose0
 * @param[in] frame1
 * @param[in] Pose1
 * @param[out] sfm_f
 */
void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4>& Pose0,
									                   int frame1, Eigen::Matrix<double, 3, 4>& Pose1, vector<SFMFeature>& sfm_f) const
{
  assert(frame0 != frame1);
  for (int j = 0; j < feature_num; j++) // 遍历所有特征点
  {
    if (sfm_f[j].state)
      continue;

    bool has_0 = false, has_1 = false;
    Vector2d point0;
    Vector2d point1;
    // 遍历该特征点的观测，是否两帧都可以看到
    for (auto& observation: sfm_f[j].observation)
    {
      if (observation.first == frame0) // 该特征点在frame0中有观测
      {
        point0 = observation.second;
        has_0 = true;
      }
      if (observation.first == frame1)
      {
        point1 = observation.second;
        has_1 = true;
      }
    }

    // 两帧都可以看到该特征点，进行三角化
    if (has_0 && has_1)
    {
      Vector3d point_3d;
      triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
      sfm_f[j].state = true;
      sfm_f[j].position[0] = point_3d(0);
      sfm_f[j].position[1] = point_3d(1);
      sfm_f[j].position[2] = point_3d(2);
    }
  }
}

// 	 q w_R_cam t w_R_cam
//  c_rotation cam_R_w 
//  c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)
/**
 * @brief 根据已有的枢纽帧和最后一帧的位姿变换，得到各帧的位姿和特征点的三维坐标，最后进行全局BA
 * @param frame_num 滑窗内的帧数
 * @param q 各帧的旋转位姿
 * @param T 各帧的平移位姿
 * @param l 枢纽帧的id
 * @param relative_R 枢纽帧和最后一帧的旋转变换
 * @param relative_T 枢纽帧和最后一帧的平移变换
 * @param sfm_f 用来做sfm的特征点集合
 * @param sfm_tracked_points 恢复出来的特征点的三维坐标
 * @return
 */
bool GlobalSFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			                    const Matrix3d& relative_R, const Vector3d& relative_T,
			                    vector<SFMFeature>& sfm_f, map<int, Vector3d>& sfm_tracked_points)
{
  feature_num = (int)sfm_f.size();
  // 枢纽帧设置为单位四元数，平移为0，也可以理解为世界坐标系的原点
  q[l].w() = 1;
  q[l].x() = 0;
  q[l].y() = 0;
  q[l].z() = 0;
  T[l].setZero();
  q[frame_num - 1] = q[l] * Quaterniond(relative_R);
  T[frame_num - 1] = relative_T;

  // 由于纯视觉slam处理的都是Tcw，因此这里需要将Twc转换成Tcw
  //rotate to cam frame
  Matrix3d c_Rotation[frame_num];
  Vector3d c_Translation[frame_num];
  Quaterniond c_Quat[frame_num];
  double c_rotation[frame_num][4];
  double c_translation[frame_num][3];
  Eigen::Matrix<double, 3, 4> Pose[frame_num];

  // 将枢纽帧和最后一帧的Twc转换成Tcw，包括四元数、旋转矩阵、平移向量和增广矩阵
  c_Quat[l] = q[l].inverse();
  c_Rotation[l] = c_Quat[l].toRotationMatrix();
  c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
  Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
  Pose[l].block<3, 1>(0, 3) = c_Translation[l];

  c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
  c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
  c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
  Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
  Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];


  // 1: triangulate between l ----- frame_num - 1
  // 2: solve pnp l + 1; triangulate l + 1 ------- frame_num - 1;
  // step1: 求解从枢纽帧到最后一帧的位姿以及特征点的三维坐标
  for (int i = l; i < frame_num - 1 ; i++)
  {
    // solve pnp
    if (i > l)
    {
      Matrix3d R_initial = c_Rotation[i - 1];
      Vector3d P_initial = c_Translation[i - 1];
      if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
        return false;
      c_Rotation[i] = R_initial;
      c_Translation[i] = P_initial;
      c_Quat[i] = c_Rotation[i];
      Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
      Pose[i].block<3, 1>(0, 3) = c_Translation[i];
    }

    // triangulate point based on the solve pnp result，每一次都会三角化一批特征点
    triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f); // 三角化特征点，特征点是在world坐标系下的
  }

  //3: triangulate l-----l+1 l+2 ... frame_num -2
  // step2: 考虑到有些特征点不能被最后一帧看到，因此，固定枢纽帧，遍历枢纽帧到倒数第2帧进行特征点三角化
  for (int i = l + 1; i < frame_num - 1; i++)
    triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
  //4: solve pnp l-1; triangulate l-1 ----- l
  //             l-2              l-2 ----- l

 // step3: 开始处理枢纽帧之前的帧，从枢纽帧开始，反向遍历到第0帧，求解各帧的位姿和特征点的三维坐标
  for (int i = l - 1; i >= 0; i--)
  {
    //solve pnp
    Matrix3d R_initial = c_Rotation[i + 1];
    Vector3d P_initial = c_Translation[i + 1];
    if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
      return false;
    c_Rotation[i] = R_initial;
    c_Translation[i] = P_initial;
    c_Quat[i] = c_Rotation[i];
    Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
    Pose[i].block<3, 1>(0, 3) = c_Translation[i];
    //triangulate
    triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
  }

  //5: triangulate all other points
  // step4: 得到了所有关键帧的位姿，但是有些特征点可能没有被三角化，因此，遍历所有特征点，对没有三角化的特征点进行三角化
  for (int j = 0; j < feature_num; j++)
  {
    if (sfm_f[j].state)
      continue;
    if ((int)sfm_f[j].observation.size() >= 2)
    {
      Vector2d point0, point1;
      // 取首尾两个KF，保证两个KF之间的视差足够大
      int frame_0 = sfm_f[j].observation[0].first;
      point0 = sfm_f[j].observation[0].second;
      int frame_1 = sfm_f[j].observation.back().first;
      point1 = sfm_f[j].observation.back().second;
      Vector3d point_3d;
      triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
      sfm_f[j].state = true;
      sfm_f[j].position[0] = point_3d(0);
      sfm_f[j].position[1] = point_3d(1);
      sfm_f[j].position[2] = point_3d(2);
    }
  }

  //full BA
  // step5: 进行全局BA，优化所有关键帧的位姿以及所有特征点的三维坐标
  ceres::Problem problem; // 创建一个优化问题
  ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization(); // 局部参数的更新方式
  //cout << " begin full BA " << endl;
  for (int i = 0; i < frame_num; i++)
  {
    //double array for ceres
    // 待优化的参数块以double数组的形式存储
    c_translation[i][0] = c_Translation[i].x();
    c_translation[i][1] = c_Translation[i].y();
    c_translation[i][2] = c_Translation[i].z();
    c_rotation[i][0] = c_Quat[i].w(); // ceres的四元数存储顺序为[w, x, y, z]
    c_rotation[i][1] = c_Quat[i].x();
    c_rotation[i][2] = c_Quat[i].y();
    c_rotation[i][3] = c_Quat[i].z();
    problem.AddParameterBlock(c_rotation[i], 4, local_parameterization); // 添加待优化的参数块
    problem.AddParameterBlock(c_translation[i], 3);
    // 由于单目slam有7个自由度不可观，因此，固定一些参数块以避免在零空间漂移
    // fix设置的是世界坐标系第l帧的位姿，同时fix最后一帧的位移用来fix尺度
    if (i == l)
    {
      problem.SetParameterBlockConstant(c_rotation[i]);
    }
    if (i == l || i == frame_num - 1)
    {
      problem.SetParameterBlockConstant(c_translation[i]);
    }
  }

  for (int i = 0; i < feature_num; i++)
  {
    if (!sfm_f[i].state)
      continue;
    for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
    {
      int L = sfm_f[i].observation[j].first;
      ceres::CostFunction* cost_function = ReprojectionError3D::Create(sfm_f[i].observation[j].second.x(),
                                                                       sfm_f[i].observation[j].second.y());

      problem.AddResidualBlock(cost_function, nullptr, c_rotation[L], c_translation[L], sfm_f[i].position);
    }

  }
  ceres::Solver::Options options; // 配置求解器的选项
  options.linear_solver_type = ceres::DENSE_SCHUR;
  //options.minimizer_progress_to_stdout = true;
  options.max_solver_time_in_seconds = 0.2;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  //std::cout << summary.BriefReport() << "\n";
  if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
  {
    cout << "vision only BA converge" << endl;
  }
  else
  {
    cout << "vision only BA not converge " << endl;
    return false;
  }

  // 优化结束后，把double数组里的结果转换为对应类型的值，同时转换将Tcw转换成Twc
  for (int i = 0; i < frame_num; i++)
  {
    q[i].w() = c_rotation[i][0];
    q[i].x() = c_rotation[i][1];
    q[i].y() = c_rotation[i][2];
    q[i].z() = c_rotation[i][3];
    q[i] = q[i].inverse();
      //cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
  }
  for (int i = 0; i < frame_num; i++)
  {

    T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
    //cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
  }
  for (auto& i: sfm_f)
  {
    if(i.state)
      sfm_tracked_points[i.id] = Vector3d(i.position[0], i.position[1], i.position[2]);
  }
  return true;

}

