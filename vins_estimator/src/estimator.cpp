#include "estimator.h"

#include <utility>

ofstream ofs("/home/zhang/vins_estimator.txt", ios::app);
Estimator::Estimator(): f_manager{Rs}
{
  ROS_INFO("init begins");
  clearState();
}

/**
 * @brief 外参，重投影置信度，延时设置
 */
void Estimator::setParameter()
{
  for (int i = 0; i < NUM_OF_CAM; i++)
  {
    tic[i] = TIC[i]; // 可能是多相机，所以外参存在一个vector里
    ric[i] = RIC[i];
  }
  f_manager.setRic(ric); // 将旋转参数设置到特征点管理器
  // 视觉特征点的置信度1.5个像素
  ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
  ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
  td = TD;
}

void Estimator::clearState()
{
  for (int i = 0; i < WINDOW_SIZE + 1; i++)
  {
    Rs[i].setIdentity();
    Ps[i].setZero();
    Vs[i].setZero();
    Bas[i].setZero();
    Bgs[i].setZero();
    dt_buf[i].clear();
    linear_acceleration_buf[i].clear();
    angular_velocity_buf[i].clear();

    if (pre_integrations[i] != nullptr)
      delete pre_integrations[i];
    pre_integrations[i] = nullptr;
  }

  for (int i = 0; i < NUM_OF_CAM; i++)
  {
    tic[i] = Vector3d::Zero();
    ric[i] = Matrix3d::Identity();
  }

  for (auto &it: all_image_frame)
  {
    if (it.second.pre_integration != nullptr)
    {
      delete it.second.pre_integration;
      it.second.pre_integration = nullptr;
    }
  }

  solver_flag = INITIAL;
  first_imu = false,
  sum_of_back = 0;
  sum_of_front = 0;
  frame_count = 0;
  initial_timestamp = 0;
  all_image_frame.clear();
  td = TD;

  delete tmp_pre_integration;
  delete last_marginalization_info;

  tmp_pre_integration = nullptr;
  last_marginalization_info = nullptr;
  last_marginalization_parameter_blocks.clear();

  f_manager.clearState();

  failure_occur = false;
  relocalization_info = false;

  drift_correct_r = Matrix3d::Identity();
  drift_correct_t = Vector3d::Zero();
}

void Estimator::processIMU(double dt, const Vector3d& linear_acceleration, const Vector3d& angular_velocity)
{
  if (!first_imu) // 第一帧IMU数据
  {
    first_imu = true;
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
  }

  // 滑窗中保留11帧，frame_count表示现在处理第几帧，一般处理到第11帧时就不变了
  // 由于预积分是帧间约束，因此第一个预积分量是用不到的
  if (!pre_integrations[frame_count])
  {
    pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
  }
  if (frame_count != 0)
  {
    pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
    //if(solver_flag != NON_LINEAR)
    tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity); // 用于初始化，同时将imu数据保存到预积分类中

    // 保存数据
    dt_buf[frame_count].push_back(dt); // 保存这一帧图像与上一帧图像之间的imu数据
    linear_acceleration_buf[frame_count].push_back(linear_acceleration);
    angular_velocity_buf[frame_count].push_back(angular_velocity);

    // 使用imu的数据更新滑窗内的P、V、Q，为非线性优化提供初始值
    int j = frame_count;
    Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g; // Rs存的是世界坐标系到imu坐标系的旋转
    Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
    Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix(); // Rs,Ps,Vs应该是上一帧的值
    Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
    Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
    Vs[j] += dt * un_acc;
  }
  acc_0 = linear_acceleration; // 每次运行完的值都是等于图像当前图像时刻的值或者是最近的大于当前图像时刻的值
  gyr_0 = angular_velocity;
}

void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& image, const std_msgs::Header& header)
{
  ROS_DEBUG("new image coming ------------------------------------------");
  ROS_DEBUG("Adding feature points %lu", image.size());
  // step1: 将特征点信息添加到特征点管理器中，同时进行上一帧是否是关键帧的检查
  if (f_manager.addFeatureCheckParallax(frame_count, image, td))
    // 如果上一帧是KF，那么滑窗内最老的帧就要被边缘化
    marginalization_flag = MARGIN_OLD;
  else
    // 否则移除上一帧
    marginalization_flag = MARGIN_SECOND_NEW;

  ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
  ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
  ROS_DEBUG("Solving %d", frame_count);
  ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
  Headers[frame_count] = header; // 保存滑窗内图像帧的时间戳

  ImageFrame imageframe(image, header.stamp.toSec()); // 初始化做图像IMU对齐的类，保存了图像帧的信息
  imageframe.pre_integration = tmp_pre_integration; // 将前面构造好的预积分类赋给当前帧，滑窗内的第一帧的预积分是用不到的，因此是nullptr
  // stamp->imageFrame
  // all_image_frame用来做初始化相关的操作，保存了滑窗内起始到当前的所有的图像帧信息
  // 有一些帧因为不是KF，被MARGIN_SECOND_NEW移除了，但是还是保存在all_image_frame中，因为初始化要求使用所有的图像帧，而非只要KF
  all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));
  tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]}; // 创建新的预积分对象，用于计算这一帧与下一帧之间的预积分

  // 没有外参初值
  // step2: 外参初始化
  if(ESTIMATE_EXTRINSIC == 2)
  {
    ROS_INFO("calibrating extrinsic param, rotation movement is needed");
    if (frame_count != 0)
    {
      vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
      Matrix3d calib_ric;
      if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
      {
        ROS_WARN("initial extrinsic rotation calib success");
        ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
        ric[0] = calib_ric;
        RIC[0] = calib_ric;
        // 标志位设置为可信的外参初值
        ESTIMATE_EXTRINSIC = 1;
      }
    }
  }

  if (solver_flag == INITIAL)
  {
    if (frame_count == WINDOW_SIZE) // 滑窗内有11帧图像，有足够的帧数
    {
      bool result = false;
      // 希望有一个比较可信的外参初值以及距离上一次初始化不成功的时间间隔大于0.1s
      if (ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
      {
        result = initialStructure();
        initial_timestamp = header.stamp.toSec();
      }
      // step3: 初始化成功，开始非线性优化
      if (result)
      {
        solver_flag = NON_LINEAR;
        // step4: 非线性优化求解VIO
        solveOdometry();
        // step5: 滑动窗口
        slideWindow();
        // step6: 移除失败的特征点
        f_manager.removeFailures(); // 移除三角化失败的特征点
        ROS_INFO("Initialization finish!");
        last_R = Rs[WINDOW_SIZE]; // 保存最新的位姿
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0]; // 保存最老的位姿
        last_P0 = Ps[0];
      }
      else
        slideWindow();
    }
    else
      frame_count++;
  }
  else
  {
    TicToc t_solve;
    solveOdometry();
    ROS_DEBUG("solver costs: %fms", t_solve.toc());

    if (failureDetection()) // 检测VIO是否正常
    {
      ROS_WARN("failure detection!");
      failure_occur = true;
      clearState();
      setParameter();
      ROS_WARN("system reboot!");
      return;
    }

    TicToc t_margin;
    slideWindow();
    f_manager.removeFailures();
    ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
    // prepare output of VINS
    // 给可视化用的
    key_poses.clear();
    for (int i = 0; i <= WINDOW_SIZE; i++)
      key_poses.push_back(Ps[i]);

    last_R = Rs[WINDOW_SIZE];
    last_P = Ps[WINDOW_SIZE];
    last_R0 = Rs[0];
    last_P0 = Ps[0];
  }
}

/**
 * @brief VIO初始化，将滑窗内的P、V和Q恢复到第0帧并且和重力进行对齐
 * @return bool
 */
bool Estimator::initialStructure()
{
  TicToc t_sfm;

  // step1: check imu observability
  {
    map<double, ImageFrame>::iterator frame_it;
    Vector3d sum_g{0, 0, 0};
    // 从第2帧开始检查imu的运动激励是否足够
    for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
    {
      double dt = frame_it->second.pre_integration->sum_dt;
      Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt; // 两个图像帧之间的平均速度
      sum_g += tmp_g;
    }
    Vector3d aver_g;
    aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1); // 滑窗内所有帧的平均加速度

    // 计算运动速度的方差
    double var = 0;
    for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
    {
      double dt = frame_it->second.pre_integration->sum_dt;
      Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
      var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
    }
    var = sqrt(var / ((int)all_image_frame.size() - 1));

    if (var < 0.25)
    {
      ROS_INFO("IMU excitation not enough!");
      return false;
    }
  }

  // step2: global sfm
  // 做一个纯视觉的slam
  Quaterniond Q[frame_count + 1];
  Vector3d T[frame_count + 1];
  map<int, Vector3d> sfm_tracked_points; // 保存优化后的特征点的3D坐标
  vector<SFMFeature> sfm_f; // 保存所有的特征点的信息

  for (auto& it_per_id: f_manager.feature)
  {
    int imu_j = it_per_id.start_frame - 1; // 观测到该特征点的第一帧
    SFMFeature tmp_feature; // 用来做后续的sfm，保存了特征点所在的帧号和归一化坐标
    tmp_feature.state = false;
    tmp_feature.id = it_per_id.feature_id;
    for (auto& it_per_frame: it_per_id.feature_per_frame) // 该特征点在每一帧的观测
    {
      imu_j++;
      Vector3d pts_j = it_per_frame.point; // 归一化平面坐标
      tmp_feature.observation.emplace_back(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()});
    }
    sfm_f.push_back(tmp_feature); // sfm_f每一个元素是一个特征点的信息
  }

  Matrix3d relative_R;
  Vector3d relative_T;
  int l;
  // 找枢纽帧
  if (!relativePose(relative_R, relative_T, l))
  {
    ROS_INFO("Not enough features or parallax; Move device around");
    return false;
  }

  GlobalSFM sfm;
  if (!sfm.construct(frame_count + 1, Q, T, l,
                     relative_R, relative_T,
                     sfm_f, sfm_tracked_points))
  {
    ROS_DEBUG("global SFM failed!");
    marginalization_flag = MARGIN_OLD;
    return false;
  }

  // step3: solve pnp for all frame
  // step2只是针对KF做了sfm，初始化需要all_image_frame中的所有帧，因此下面通过KF来求解其他的非KF的帧的位姿
  map<double, ImageFrame>::iterator frame_it;
  map<int, Vector3d>::iterator it;
  frame_it = all_image_frame.begin();
  for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
  {
    // provide initial guess
    cv::Mat r, rvec, t, D, tmp_r;
    if ((frame_it->first) == Headers[i].stamp.toSec()) // 如果时间戳相等，说明是KF
    {
      frame_it->second.is_key_frame = true;
      frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose(); // Twi
      frame_it->second.T = T[i]; // 初始化不估计平移外参
      i++;
      continue;
    }
    if ((frame_it->first) > Headers[i].stamp.toSec()) // 偏移i，找到最接近的KF
    {
      i++;
    }

    // 最近的KF提供初始值，Twc->Tcw
    Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
    Vector3d P_inital = -R_inital * T[i];
    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    frame_it->second.is_key_frame = false;
    vector<cv::Point3f> pts_3_vector;
    vector<cv::Point2f> pts_2_vector;
    for (auto& id_pts: frame_it->second.points)
    {
      int feature_id = id_pts.first;
      for (auto& i_p: id_pts.second)
      {
        it = sfm_tracked_points.find(feature_id);
        if(it != sfm_tracked_points.end()) // 如果该特征点在sfm中有对应的3D点
        {
          Vector3d world_pts = it->second;
          cv::Point3f pts_3((float)world_pts(0), (float)world_pts(1), (float)world_pts(2));
          pts_3_vector.push_back(pts_3);
          Vector2d img_pts = i_p.second.head<2>();
          cv::Point2f pts_2((float)img_pts(0), (float)img_pts(1));
          pts_2_vector.push_back(pts_2);
        }
      }
    }

    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    if (pts_3_vector.size() < 6)
    {
      cout << "pts_3_vector size " << pts_3_vector.size() << endl;
      ROS_DEBUG("Not enough points for solve pnp !");
      return false;
    }
    if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, true))
    {
      ROS_DEBUG("solve pnp fail!");
      return false;
    }
    cv::Rodrigues(rvec, r);
    // cv->eigen，同时Tcw->Twc
    MatrixXd R_pnp,tmp_R_pnp;
    cv::cv2eigen(r, tmp_R_pnp);
    R_pnp = tmp_R_pnp.transpose();
    MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);
    T_pnp = R_pnp * (-T_pnp);
    // Twc->Twi，由于尺度未恢复，因此平移暂不转到imu系
    frame_it->second.R = R_pnp * RIC[0].transpose();
    frame_it->second.T = T_pnp;
  }

  // step4: 视觉惯性对齐
  if (visualInitialAlign())
    return true;
  else
  {
    ROS_INFO("misaligned visual structure with IMU");
    return false;
  }

}

bool Estimator::visualInitialAlign()
{
  TicToc t_g;
  VectorXd x;

  // solve scale
  bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
  if (!result)
  {
    ROS_DEBUG("solve g failed!");
    return false;
  }

  // change state
  // 首先把对齐后的KF位姿赋给滑窗内的值，Twc，twc
  for (int i = 0; i <= frame_count; i++)
  {
    Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R;
    Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T;
    Ps[i] = Pi;
    Rs[i] = Ri;
    all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
  }

  VectorXd dep = f_manager.getDepthVector(); // 根据特征点数初始化特征点逆深度向量
  for (int i = 0; i < dep.size(); i++)
    dep[i] = -1;
  f_manager.clearDepth(dep);

  // triangulation on cam pose , no tic
  Vector3d TIC_TMP[NUM_OF_CAM];
  for(auto& i: TIC_TMP)
    i.setZero();
  ric[0] = RIC[0]; // imu到相机的旋转
  f_manager.setRic(ric);
  // 多约束三角化所有特征点，仍是带有尺度模糊
  f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

  double s = (x.tail<1>())(0);
  // 重新计算滑窗内的预积分
  for (int i = 0; i <= WINDOW_SIZE; i++)
  {
    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
  }
  // 下面把所有状态量对齐到第0帧的imu坐标系
  for (int i = frame_count; i >= 0; i--) // Rs是枢纽帧->imu
    // twi - tw0 = t0i（枢纽帧坐标系），把所有的平移对齐到滑窗的第0帧，Ps还是在枢纽帧坐标系下
    Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]); // 相当于把世界系的原点移到了第0帧的imu坐标系的原点

  //将求解出来的速度赋值给滑窗内的状态量
  int kv = -1;
  map<double, ImageFrame>::iterator frame_i;
  for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
  {
    if(frame_i->second.is_key_frame)
    {
      kv++;
      // 当时求得的速度是imu系下的，需要转到枢纽帧坐标系下
      Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
    }
  }

  // 将尺度模糊3d特征点恢复到真实尺度
  for (auto& it_per_id: f_manager.feature)
  {
    it_per_id.used_num = (int)it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;
    it_per_id.estimated_depth *= s; // 深度是在相机坐标系
  }

  // 所有的P V Q全部对齐到第0帧，同时将第0帧与重力方向对齐
  Matrix3d R0 = Utility::g2R(g); // g是枢纽帧坐标系下的重力，得到Rw_sn
  double yaw = Utility::R2ypr(R0 * Rs[0]).x(); // 第0帧的yaw角（相对于真正的world系）
  R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0; // 得到的R0是对齐到重力方向下的，同时yaw角对齐到第0帧
  g = R0 * g;

  // Matrix3d rot_diff = R0 * Rs[0].transpose();
  Matrix3d rot_diff = R0;
  for (int i = 0; i <= frame_count; i++)
  {
    Ps[i] = rot_diff * Ps[i]; // 对齐到重力方向下，同时yaw角对齐到第0帧
    Rs[i] = rot_diff * Rs[i];
    Vs[i] = rot_diff * Vs[i];
  }
  ROS_DEBUG_STREAM("g0     " << g.transpose());
  ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

  return true;
}

bool Estimator::relativePose(Matrix3d& relative_R, Vector3d& relative_T, int& l)
{
  // find previous frame which contains enough correspondence and parallax with the newest frame
  for (int i = 0; i < WINDOW_SIZE; i++)
  {
    vector<pair<Vector3d, Vector3d>> corres;
    corres = f_manager.getCorresponding(i, WINDOW_SIZE);

    // 要求共视的特征点数目大于20个（论文中写的是30）
    if (corres.size() > 20)
    {
      double sum_parallax = 0;
      double average_parallax;
      for (auto& corre: corres)
      {
        Vector2d pts_0(corre.first(0), corre.first(1));
        Vector2d pts_1(corre.second(0), corre.second(1));
        double parallax = (pts_0 - pts_1).norm();
        sum_parallax += parallax;
      }
      average_parallax = 1.0 * sum_parallax / int(corres.size());

      // 如果i帧与滑窗内最后一帧的平均视差大于30个像素，同时求解得到相对位姿满足条件
      if(average_parallax * 460 > 30 && MotionEstimator::solveRelativeRT(corres, relative_R, relative_T))
      {
        l = i; // 将第i帧设置为枢纽帧
        ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
        return true;
      }
    }
  }

  return false;
}

void Estimator::solveOdometry()
{
  if (frame_count < WINDOW_SIZE) // 保证滑窗内帧数满了
    return;
  if (solver_flag == NON_LINEAR)
  {
    TicToc t_tri;
    // 先把应该三角化但是没有三角化的特征点三角化
    f_manager.triangulate(Ps, tic, ric);
    ROS_DEBUG("triangulation costs %f", t_tri.toc());
    optimization();
  }
}

/**
 * @brief 由于ceres的参数块都是double数组，因此这里需要将状态量从eigen转换成double数组
 */
void Estimator::vector2double()
{
  for (int i = 0; i <= WINDOW_SIZE; i++)
  {
    para_Pose[i][0] = Ps[i].x();
    para_Pose[i][1] = Ps[i].y();
    para_Pose[i][2] = Ps[i].z();
    Quaterniond q{Rs[i]};
    para_Pose[i][3] = q.x();
    para_Pose[i][4] = q.y();
    para_Pose[i][5] = q.z();
    para_Pose[i][6] = q.w();

    para_SpeedBias[i][0] = Vs[i].x();
    para_SpeedBias[i][1] = Vs[i].y();
    para_SpeedBias[i][2] = Vs[i].z();

    para_SpeedBias[i][3] = Bas[i].x();
    para_SpeedBias[i][4] = Bas[i].y();
    para_SpeedBias[i][5] = Bas[i].z();

    para_SpeedBias[i][6] = Bgs[i].x();
    para_SpeedBias[i][7] = Bgs[i].y();
    para_SpeedBias[i][8] = Bgs[i].z();
  }

  for (int i = 0; i < NUM_OF_CAM; i++)
  {
    para_Ex_Pose[i][0] = tic[i].x();
    para_Ex_Pose[i][1] = tic[i].y();
    para_Ex_Pose[i][2] = tic[i].z();
    Quaterniond q{ric[i]};
    para_Ex_Pose[i][3] = q.x();
    para_Ex_Pose[i][4] = q.y();
    para_Ex_Pose[i][5] = q.z();
    para_Ex_Pose[i][6] = q.w();
  }

  VectorXd dep = f_manager.getDepthVector();
  for (int i = 0; i < f_manager.getFeatureCount(); i++)
    para_Feature[i][0] = dep(i);
  if (ESTIMATE_TD)
    para_Td[0][0] = td;
}

/**
 * @brief double->eigen 同时fix第0帧的yaw和平移，固定了四自由度的零空间
 */
void Estimator::double2vector()
{
  // 取出优化前的第i帧的位姿
  Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
  Vector3d origin_P0 = Ps[0];

  if (failure_occur)
  {
    origin_R0 = Utility::R2ypr(last_R0);
    origin_P0 = last_P0;
    failure_occur = false;
  }
  // 取出优化后的第i帧的位姿
  Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6], para_Pose[0][3],
                                                      para_Pose[0][4],para_Pose[0][5]).toRotationMatrix());
  double y_diff = origin_R0.x() - origin_R00.x();
  //TODO
  Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
  // 接近奇异
  if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
  {
    ROS_DEBUG("euler singular point!");
    rot_diff = Rs[0] * Quaterniond(para_Pose[0][6], para_Pose[0][3],para_Pose[0][4],
                                   para_Pose[0][5]).toRotationMatrix().transpose();
  }

  for (int i = 0; i <= WINDOW_SIZE; i++)
  {
    // 保证第1帧的yaw不变
    Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

    Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

    Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],para_SpeedBias[i][1],para_SpeedBias[i][2]);

    Bas[i] = Vector3d(para_SpeedBias[i][3],para_SpeedBias[i][4],para_SpeedBias[i][5]);

    Bgs[i] = Vector3d(para_SpeedBias[i][6],para_SpeedBias[i][7],para_SpeedBias[i][8]);
  }

  for (int i = 0; i < NUM_OF_CAM; i++)
  {
    tic[i] = Vector3d(para_Ex_Pose[i][0],para_Ex_Pose[i][1],para_Ex_Pose[i][2]);
    ric[i] = Quaterniond(para_Ex_Pose[i][6],para_Ex_Pose[i][3],para_Ex_Pose[i][4],para_Ex_Pose[i][5]).toRotationMatrix();
  }

  VectorXd dep = f_manager.getDepthVector();
  for (int i = 0; i < f_manager.getFeatureCount(); i++)
    dep(i) = para_Feature[i][0];
  f_manager.setDepth(dep);
  if (ESTIMATE_TD)
    td = para_Td[0][0];

  // relative info between two loop frame
  if(relocalization_info) // 类似进行一个调整
  {
    Matrix3d relo_r;
    Vector3d relo_t;
    relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
    relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],relo_Pose[1] - para_Pose[0][1],
                                 relo_Pose[2] - para_Pose[0][2]) + origin_P0;
    double drift_correct_yaw;
    drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
    drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
    drift_correct_t = prev_relo_t - drift_correct_r * relo_t;
    // T_loop_cur = T_loop_w * T_w_cur
    relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
    relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
    relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
    //cout << "vins relo " << endl;
    //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
    //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
    relocalization_info = false;
  }
}

bool Estimator::failureDetection()
{
  if (f_manager.last_track_num < 2) // 地图点数目是否足够
  {
    ROS_INFO(" little feature %d", f_manager.last_track_num);
    //return true;
  }
  if (Bas[WINDOW_SIZE].norm() > 2.5) // 估计的加速度偏置是否过大
  {
    ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
    return true;
  }
  if (Bgs[WINDOW_SIZE].norm() > 1.0) // 估计的陀螺仪偏置是否过大
  {
    ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
    return true;
  }
  /*
  if (tic(0) > 1)
  {
      ROS_INFO(" big extra param estimation %d", tic(0) > 1);
      return true;
  }
  */
  Vector3d tmp_P = Ps[WINDOW_SIZE];
  if ((tmp_P - last_P).norm() > 5) // 两帧之间的运动是否过大
  {
    ROS_INFO(" big translation");
    return true;
  }
  if (abs(tmp_P.z() - last_P.z()) > 1) // z方向的运动是否过大
  {
    ROS_INFO(" big z translation");
    return true;
  }
  Matrix3d tmp_R = Rs[WINDOW_SIZE];
  Matrix3d delta_R = tmp_R.transpose() * last_R;
  Quaterniond delta_Q(delta_R);
  double delta_angle;
  delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
  if (delta_angle > 50) // 两帧之间的旋转是否过大
  {
    ROS_INFO(" big delta_angle ");
    //return true;
  }
  return false;
}

/**
 * @brief 进行滑窗非线性优化
 */
void Estimator::optimization()
{
  // 借助ceres进行非线性优化
  ceres::Problem problem;
  ceres::LossFunction* loss_function; // 核函数
  // loss_function = new ceres::HuberLoss(1.0); a控制分段函数的分段点
  // CauchyLoss通过参数a来控制核函数的增长速度，a越大，增长越快，对outlier容忍性越低
  loss_function = new ceres::CauchyLoss(1.0); // Cauchy鲁棒核函数, pho(s) = a ** 2 * log(1 + (s / a ** 2))

  // step1： 定义待优化的参数块
  // 参数块1 滑窗中的位姿包括位置和姿态，维度为7（更新方式不是广义加的需要以定义添加参数块方式指定更新方法）
  for (int i = 0; i < WINDOW_SIZE + 1; i++)
  {
    ceres::LocalParameterization* local_parameterization = new PoseLocalParameterization(); // 自定义的位姿更新方式
    problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
    problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
  }

  for (auto& i: para_Ex_Pose)
  {
    ceres::LocalParameterization* local_parameterization = new PoseLocalParameterization();
    problem.AddParameterBlock(i, SIZE_POSE, local_parameterization);
    if (!ESTIMATE_EXTRINSIC)
    {
      ROS_DEBUG("fix extrinsic param");
      problem.SetParameterBlockConstant(i);
    }
    else
      ROS_DEBUG("estimate extrinsic param");
  }

  if (ESTIMATE_TD)
  {
    problem.AddParameterBlock(para_Td[0], 1);
  }

  TicToc t_whole, t_prepare;
  // eigen->double数组
  vector2double();

  // step2: 通过残差的约束添加残差块，类似于G2O中的edge
  // 上一次边缘化结果作为这一次的先验
  if (last_marginalization_info)
  {
    // construct new marginalization_factor
    auto* marginalization_factor = new MarginalizationFactor(last_marginalization_info);
    problem.AddResidualBlock(marginalization_factor, nullptr,
                             last_marginalization_parameter_blocks);
  }

  // imu预积分的结果
  for (int i = 0; i < WINDOW_SIZE; i++)
  {
    int j = i + 1;
    if (pre_integrations[j]->sum_dt > 10.0) // 时间过长这个约束就不可信了，不添加约束
      continue;
    auto* imu_factor = new IMUFactor(pre_integrations[j]);
    problem.AddResidualBlock(imu_factor, nullptr, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
  }

  // 视觉重投影的约束，首先遍历所有特征点
  int f_m_cnt = 0;
  int feature_index = -1;
  for (auto& it_per_id: f_manager.feature)
  {
    it_per_id.used_num = (int)it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;

    ++feature_index;
    // 第一个观测到该特征点的帧的id
    int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

    Vector3d pts_i = it_per_id.feature_per_frame[0].point; // 第一个观测到该特征点的相机坐标系下的归一化坐标

    // 找到该特征点被哪些KF看到
    for (auto& it_per_frame: it_per_id.feature_per_frame)
    {
      imu_j++;
      if (imu_i == imu_j) // 自己和自己不能形成重投影约束
      {
        continue;
      }
      Vector3d pts_j = it_per_frame.point;
      if (ESTIMATE_TD)
      {
        auto* f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                            it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                            it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
        problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
      }
      else
      {
        auto* f = new ProjectionFactor(pts_i, pts_j); // 构造函数是同一个特征点在不同帧的观测
        // 约束变量是该特征点的第一个观测帧以及另一个观测帧，加上外参和特征点的逆深度
        problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
      }
      f_m_cnt++;
    }
  }

  ROS_DEBUG("visual measurement count: %d", f_m_cnt);
  ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

  // 回环检测的约束
  if(relocalization_info)
  {
    //printf("set relocalization factor! \n");
    ceres::LocalParameterization* local_parameterization = new PoseLocalParameterization();
    problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization); // 优化变量是重定位帧的位姿
    int retrive_feature_index = 0;
    int feature_index = -1;
    // 遍历所有的特征点
    for (auto& it_per_id: f_manager.feature)
    {
      it_per_id.used_num = (int)it_per_id.feature_per_frame.size();
      if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
        continue;
      ++feature_index;
      int start = it_per_id.start_frame;
      if(start <= relo_frame_local_index) // 这个地图点被对应的当前帧看到
      {
        // 寻找回环可以看到的地图点
        while((int)match_points[retrive_feature_index].z() < it_per_id.feature_id)
        {
          retrive_feature_index++;
        }
        // 这个地图点也可以被回环帧看到
        if((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
        {
          // 构造重定位的重投影约束，约束的是当前帧和回环帧
          Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
          Vector3d pts_i = it_per_id.feature_per_frame[0].point;

          auto* f = new ProjectionFactor(pts_i, pts_j);
          problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
          retrive_feature_index++;
        }
      }
    }

  }

  ceres::Solver::Options options;

  options.linear_solver_type = ceres::DENSE_SCHUR;
  //options.num_threads = 2;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.max_num_iterations = NUM_ITERATIONS;
  if (marginalization_flag == MARGIN_OLD)
    options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
  else
    options.max_solver_time_in_seconds = SOLVER_TIME;
  TicToc t_solver;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  //cout << summary.BriefReport() << endl;
  ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
  ROS_DEBUG("solver costs: %f", t_solver.toc());

  // 把优化后的double数组转换成eigen
  double2vector();

  TicToc t_whole_marginalization;
  if (marginalization_flag == MARGIN_OLD)
  { // 一个用来边缘化操作的对象
    auto* marginalization_info = new MarginalizationInfo();
    vector2double(); // 这里类似手写高斯牛顿，因此也需要都转成double数组

    // 关于边缘化有几点注意的地方
    // 1、找到需要边缘化的参数块，这里是地图点，第0帧位姿，第0帧速度零偏
    // 2、找到构造高斯牛顿下降时跟这些待边缘化相关的参数块有关的残差约束，那就是预积分约束，重投影约束，以及上一次边缘化约束
    // 3、这些约束连接的参数块中，不需要被边缘化的参数块，就是被提供先验约束的部分，也就是滑窗中剩下的位姿和速度零偏
    if (last_marginalization_info) // 上一次的边缘化结果
    {
      vector<int> drop_set;
      // last_marginalization_parameter_blocks是上一次边缘化对哪些当前参数块有约束
      for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
      {
        // 涉及到的待边缘化的上一次边缘化留下来的当前参数块只有位姿和速度零偏
        if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
            last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
          drop_set.push_back(i);
      }
      // construct new marginalization_factor
      auto* marginalization_factor = new MarginalizationFactor(last_marginalization_info);
      auto* residual_block_info = new ResidualBlockInfo(marginalization_factor, nullptr,
                                                        last_marginalization_parameter_blocks, drop_set);

      marginalization_info->addResidualBlockInfo(residual_block_info);
    }

    // 只有第1个预积分和待边缘化参数块相连
    {
      if (pre_integrations[1]->sum_dt < 10.0)
      {
        auto* imu_factor = new IMUFactor(pre_integrations[1]);
        auto *residual_block_info = new ResidualBlockInfo(imu_factor, nullptr,vector<double *>{para_Pose[0],
                                                           para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},vector<int>{0, 1});
        marginalization_info->addResidualBlockInfo(residual_block_info);                                 // 这里就是第0和1个参数块是需要被边缘化的
      }
    }

    // 遍历视觉重投影的约束
    {
      int feature_index = -1;
      for (auto& it_per_id: f_manager.feature)
      {
        it_per_id.used_num = (int)it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
          continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        if (imu_i != 0) // 只找能被第0帧看到的特征点
          continue;

        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        // 遍历看到这个特征点的所有KF，通过这个特征点，建立和第0帧的约束
        for (auto& it_per_frame: it_per_id.feature_per_frame)
        {
          imu_j++;
          if (imu_i == imu_j)
            continue;

          Vector3d pts_j = it_per_frame.point;
          if (ESTIMATE_TD) // 根据是否约束延时确定残差阵
          {
            auto* f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
            auto* residual_block_info = new ResidualBlockInfo(f_td, loss_function,vector<double *>{para_Pose[imu_i], para_Pose[imu_j],
                                                              para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},vector<int>{0, 3});
            marginalization_info->addResidualBlockInfo(residual_block_info);
          }
          else
          {
            auto* f = new ProjectionFactor(pts_i, pts_j);
            auto* residual_block_info = new ResidualBlockInfo(f, loss_function,vector<double *>{para_Pose[imu_i],
                                                               para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},vector<int>{0, 3});
            marginalization_info->addResidualBlockInfo(residual_block_info);
          }
        }
      }
    }

    // 所有的残差块都收集好了，进行预处理
    TicToc t_pre_margin;
    marginalization_info->preMarginalize();
    ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

    // 进行边缘化
    TicToc t_margin;
    marginalization_info->marginalize();
    ROS_DEBUG("marginalization %f ms", t_margin.toc());

    // 即将滑窗，因此记录新地址对应的老地址
    std::unordered_map<long, double*> addr_shift;
    for (int i = 1; i <= WINDOW_SIZE; i++)
    {
      // 位姿和速度都要滑窗偏移
      addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
      addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
    }

    // 外参和时间延迟不需要滑窗偏移
    for (auto& i: para_Ex_Pose)
      addr_shift[reinterpret_cast<long>(i)] = i;
    if (ESTIMATE_TD)
    {
      addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
    }
    vector<double*> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

    // 释放上一次边缘化的内存
    delete last_marginalization_info;
    last_marginalization_info = marginalization_info; // 保存这一次边缘化的结果
    last_marginalization_parameter_blocks = parameter_blocks; // 保存这一次边缘化的结果对哪些参数块有约束，这些参数块在滑窗之后的地址
  }

  else // 边缘化倒数第二帧
  { // 要求有上一次边缘化的结果，同时即将被margin掉的在上一次边缘化的约束中，预积分结果合并因此只要位姿margin掉就行
    if (last_marginalization_info && std::count(std::begin(last_marginalization_parameter_blocks),
                                                std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
    {                                                                                          // 如果上一次边缘化的结果中包含倒数第二帧的位姿
      auto* marginalization_info = new MarginalizationInfo();
      vector2double();
      if (last_marginalization_info)
      {
        vector<int> drop_set;
        for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
        {
          ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
          if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
            drop_set.push_back(i);
        }
        // construct new marginalization_factor
        auto* marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        auto* residual_block_info = new ResidualBlockInfo(marginalization_factor, nullptr,
                                                          last_marginalization_parameter_blocks, drop_set);
        marginalization_info->addResidualBlockInfo(residual_block_info);
      }

      TicToc t_pre_margin;
      ROS_DEBUG("begin marginalization");
      marginalization_info->preMarginalize();
      ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

      TicToc t_margin;
      ROS_DEBUG("begin marginalization");
      marginalization_info->marginalize();
      ROS_DEBUG("end marginalization, %f ms", t_margin.toc());

      std::unordered_map<long, double*> addr_shift;
      for (int i = 0; i <= WINDOW_SIZE; i++)
      {
        if (i == WINDOW_SIZE - 1)
          continue;
        else if (i == WINDOW_SIZE) // 滑窗，最新帧成为次新帧
        {
          addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
          addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        else
        {
          addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
          addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
        }
      }
      for (auto& i: para_Ex_Pose)
        addr_shift[reinterpret_cast<long>(i)] = i;
      if (ESTIMATE_TD)
      {
        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
      }

      vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
      delete last_marginalization_info;
      last_marginalization_info = marginalization_info;
      last_marginalization_parameter_blocks = parameter_blocks;
    }
  }
  ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());

  ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}

// 滑动窗口
void Estimator::slideWindow()
{
  TicToc t_margin;
  // 根据边缘化种类的不同，进行滑窗的方式也不同
  if (marginalization_flag == MARGIN_OLD)
  {
    double t_0 = Headers[0].stamp.toSec();
    back_R0 = Rs[0];
    back_P0 = Ps[0];

    // 必须是填满了滑窗才可以滑动
    if (frame_count == WINDOW_SIZE)
    {
      // 一帧一帧的滑动，滑动完成之后，最老帧在滑窗的位置就是最新帧的位置，其他帧依次往前移动
      for (int i = 0; i < WINDOW_SIZE; i++)
      {
        Rs[i].swap(Rs[i + 1]);

        std::swap(pre_integrations[i], pre_integrations[i + 1]);

        dt_buf[i].swap(dt_buf[i + 1]);
        linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
        angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

        Headers[i] = Headers[i + 1];
        Ps[i].swap(Ps[i + 1]);
        Vs[i].swap(Vs[i + 1]);
        Bas[i].swap(Bas[i + 1]);
        Bgs[i].swap(Bgs[i + 1]);
      }

      // 最后一帧的状态量赋上当前值
      Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
      Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
      Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
      Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
      Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
      Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

      delete pre_integrations[WINDOW_SIZE]; // 预积分量置零
      pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

      dt_buf[WINDOW_SIZE].clear();
      linear_acceleration_buf[WINDOW_SIZE].clear();
      angular_velocity_buf[WINDOW_SIZE].clear();

      // 预积分量是堆内存的指针，因此需要手动释放
      map<double, ImageFrame>::iterator it_0;
      it_0 = all_image_frame.find(t_0); // 最老帧的时间戳
      delete it_0->second.pre_integration;
      it_0->second.pre_integration = nullptr;
      for (auto it = all_image_frame.begin(); it != it_0; ++it) {
        delete it->second.pre_integration;
        it->second.pre_integration = nullptr;
      }
      all_image_frame.erase(all_image_frame.begin(), it_0);
      all_image_frame.erase(t_0);
      slideWindowOld();
    }
  }
  else
  {
    if (frame_count == WINDOW_SIZE)
    { // 将最后两个预积分量合并
      for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
      {
        double tmp_dt = dt_buf[frame_count][i];
        Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
        Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

        // 将最后一帧前的imu测量信息添加到上一个预积分的测量信息中
        pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);
        dt_buf[frame_count - 1].push_back(tmp_dt);
        linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
        angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
      }

      Headers[frame_count - 1] = Headers[frame_count];
      Ps[frame_count - 1] = Ps[frame_count];
      Vs[frame_count - 1] = Vs[frame_count];
      Rs[frame_count - 1] = Rs[frame_count];
      Bas[frame_count - 1] = Bas[frame_count];
      Bgs[frame_count - 1] = Bgs[frame_count];

      delete pre_integrations[WINDOW_SIZE];
      pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

      dt_buf[WINDOW_SIZE].clear();
      linear_acceleration_buf[WINDOW_SIZE].clear();
      angular_velocity_buf[WINDOW_SIZE].clear();

      slideWindowNew(); //
    }
  }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
  sum_of_front++;
  f_manager.removeFront(frame_count);
}

// real marginalization is removed in solve_ceres()
// 由于地图点绑定在第一个观测到它的帧上，因此需要对被移除的帧上的地图点进行解绑，以及每个地图点的收个观测id减1
void Estimator::slideWindowOld()
{
  sum_of_back++;

  bool shift_depth = solver_flag == NON_LINEAR;
  if (shift_depth) // 如果初始化完成了
  {
    Matrix3d R0, R1;
    Vector3d P0, P1;
    // back_R0是最老帧的位姿，back_P0是最老帧的位置
    R0 = back_R0 * ric[0]; // 转换为Rwc
    R1 = Rs[0] * ric[0]; // 倒数第二帧的Rwc
    P0 = back_P0 + back_R0 * tic[0];
    P1 = Ps[0] + Rs[0] * tic[0];
    // 把移除帧看见的地图点的管理权移交给当前最老帧
    f_manager.removeBackShiftDepth(R0, P0, R1, P1); // 将最老帧看到的地图点移交给次老帧（坐标变换到该帧坐标系下）
  }
  else
    f_manager.removeBack(); // 如果初始化没有完成，则将最老帧之外的其余起始帧--，同时将每个特征点在最老帧上的观测帧删掉
}

/**
 * @brief 接受回环帧的信息
 * @param[in] _frame_stamp
 * @param[in] _frame_index
 * @param[in] _match_points
 * @param[in] _relo_t
 * @param[in] _relo_r
 */
void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d>& _match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
  relo_frame_stamp = _frame_stamp;
  relo_frame_index = _frame_index;
  match_points.clear();
  match_points = _match_points;
  prev_relo_t = std::move(_relo_t);
  prev_relo_r = std::move(_relo_r);
  for(int i = 0; i < WINDOW_SIZE; i++)
  {
    if(relo_frame_stamp == Headers[i].stamp.toSec())
    {
      relo_frame_local_index = i; // 回环帧在滑窗中的位置
      relocalization_info = true; // 有效回环帧
      for (int j = 0; j < SIZE_POSE; j++)
        relo_Pose[j] = para_Pose[i][j]; // 借助VIO优化回环帧的位姿，初始值设置为当前滑窗中的位姿
    }
  }
}

