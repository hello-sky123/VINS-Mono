#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
  for (int i = 0; i < NUM_OF_CAM; i++)
    ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
  for (int i = 0; i < NUM_OF_CAM; i++)
  {
    ric[i] = _ric[i];
  }
}

void FeatureManager::clearState()
{
  feature.clear();
}

// 返回有效的特征点数目
int FeatureManager::getFeatureCount()
{
  int cnt = 0;
  for (auto& it: feature)
  {

    it.used_num = (int)it.feature_per_frame.size(); // id为it的特征点在滑窗内的观测次数

    if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
    {
      cnt++;
    }
  }
  return cnt;
}


bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& image, double td)
{
  ROS_DEBUG("input feature: %d", (int)image.size());
  ROS_DEBUG("num of feature: %d", getFeatureCount());
  double parallax_sum = 0;
  int parallax_num = 0;
  last_track_num = 0;
  for (auto& id_pts: image) // 使用范围for循环遍历map时，id_pts的类型是pair<const Key, Value>，其中Key是map的键类型，Value是map的值类型
  { // FeaturePerFrame保存了关键帧的信息
    FeaturePerFrame f_per_fra(id_pts.second[0].second, td); // 使用vector的原因是可能是双目或者单目

    int feature_id = id_pts.first;
    // 在feature中查找是否有id为feature_id的该特征点
    auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId& it)
    {
      return it.feature_id == feature_id;
    });

    if (it == feature.end()) // 如果没有找到，说明是一个新的特征点
    {
      // 在特征点管理器中，新创建一个特征点id，这里的frame_fount是该特征点在滑窗内的当前位置，作为该特征点的起始帧
      feature.emplace_back(feature_id, frame_count);
      feature.back().feature_per_frame.push_back(f_per_fra); // 将该特征点的坐标、速度信息加入到feature_per_frame中
    }
    else if (it->feature_id == feature_id)
    {
      it->feature_per_frame.push_back(f_per_fra);
      last_track_num++; // 上一帧跟踪到的特征点数
    }
  }

  if (frame_count < 2 || last_track_num < 20) // 前两帧都设置为关键帧，上一帧追踪到的特征点数目过少也设置为关键帧
    return true;

  for (auto& it_per_id: feature)
  {
    // 判断前一帧是否是关键帧，因此，起始帧至少是frame_count - 2，同时至少覆盖到frame_count - 1帧
    // it_per_id.feature_per_frame.size()表示该特征点在滑窗内的观测次数
    if (it_per_id.start_frame <= frame_count - 2 &&
      it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
    {                 // 计算parallax between second last frame and third last frame
      parallax_sum += compensatedParallax2(it_per_id, frame_count); // 总的视差
      parallax_num++; // 共视的特征点数目
    }
  }

  if (parallax_num == 0) // 和上一帧没有相同的特征点
  {
    return true;
  }
  else
  {
    ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
    ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
    return parallax_sum / parallax_num >= MIN_PARALLAX; // 平均视差是否大于阈值
  }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

/**
 * @brief 得到同时被相邻两帧观测到的特征点的归一化坐标
 * @param[in] frame_count_l
 * @param[in] frame_count_r
 * @return vector<pair<Vector3d, Vector3d>>
 */
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
  vector<pair<Vector3d, Vector3d>> corres;
  for (auto& it: feature)
  {
    // 判断该特征点是否同时被相邻两帧观测到
    if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
    {
      Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
      int idx_l = frame_count_l - it.start_frame;
      int idx_r = frame_count_r - it.start_frame;

      a = it.feature_per_frame[idx_l].point;

      b = it.feature_per_frame[idx_r].point;

      corres.emplace_back(a, b);
    }
  }
  return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::removeFailures()
{
  for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
  {
    it_next++;
    if (it->solve_flag == 2)
      feature.erase(it);
  }
}

// 把给定的深度值赋值给各个特征点作为逆深度
void FeatureManager::clearDepth(const VectorXd& x)
{
  int feature_index = -1;
  for (auto& it_per_id: feature)
  {
    it_per_id.used_num = (int)it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;
    it_per_id.estimated_depth = 1.0 / x(++feature_index);
  }
}

VectorXd FeatureManager::getDepthVector()
{
  VectorXd dep_vec(getFeatureCount());
  int feature_index = -1;
  for (auto& it_per_id: feature)
  {
    it_per_id.used_num = (int)it_per_id.feature_per_frame.size(); // used_num是id为it_per_id的特征点在滑窗内的观测次数
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;
#if 1
    dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
    dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
  }
  return dep_vec;
}

/**
 * @brief 根据观测到该特征点的所有帧来三角化特征点
 * @param[in] Ps
 * @param[in] tic
 * @param[in] ric
 */
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
  for (auto& it_per_id: feature)
  {
    it_per_id.used_num = (int)it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;

    if (it_per_id.estimated_depth > 0) // 没有三角化的话是-1
      continue;
    int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

    ROS_ASSERT(NUM_OF_CAM == 1);
    Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
    int svd_idx = 0;

    Eigen::Matrix<double, 3, 4> P0;
    // Twi->Twc，第一个观测到该特征点的KF的位姿
    Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0]; // tic预设为0
    Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
    P0.leftCols<3>() = Eigen::Matrix3d::Identity();
    P0.rightCols<1>() = Eigen::Vector3d::Zero();

    // 遍历所有观测到该特征点的KF
    for (auto& it_per_frame: it_per_id.feature_per_frame)
    {
      imu_j++;

      // 得到该KF的相机坐标系位姿
      Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
      Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
      // T_w_cj -> T_c0_cj，这里的c0是第一个观测到该特征点的KF的相机坐标系
      Eigen::Vector3d t = R0.transpose() * (t1 - t0);
      Eigen::Matrix3d R = R0.transpose() * R1;
      Eigen::Matrix<double, 3, 4> P;
      P.leftCols<3>() = R.transpose();
      P.rightCols<1>() = -R.transpose() * t;
      Eigen::Vector3d f = it_per_frame.point.normalized(); // 该帧归一化坐标
      // 构建三角化的系数矩阵
      svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
      svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

      if (imu_i == imu_j)
        continue;
    }
    ROS_ASSERT(svd_idx == svd_A.rows());
    Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
    double svd_method = svd_V[2] / svd_V[3];
    //it_per_id->estimated_depth = -b / A;
    //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

    // 得到的深度实际上是第一个观测到该特征点的KF的相机坐标系下的深度
    it_per_id.estimated_depth = svd_method;
    //it_per_id->estimated_depth = INIT_DEPTH;

    if (it_per_id.estimated_depth < 0.1)
    {
      it_per_id.estimated_depth = INIT_DEPTH;
    }

  }
}

void FeatureManager::removeOutlier()
{
  ROS_BREAK();
  int i = -1;
  for (auto it = feature.begin(), it_next = feature.begin();
    it != feature.end(); it = it_next)
  {
    it_next++;
    i += it->used_num != 0;
    if (it->used_num != 0 && it->is_outlier)
    {
      feature.erase(it);
    }
  }
}

void FeatureManager::removeBackShiftDepth(const Eigen::Matrix3d& marg_R, const Eigen::Vector3d& marg_P,
                                          const Eigen::Matrix3d& new_R, const Eigen::Vector3d& new_P)
{
  for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
  {
    it_next++;

    if (it->start_frame != 0) // 如果不是被移除帧看到的，那么地图点的起始帧减1
      it->start_frame--;
    else
    {
      Eigen::Vector3d uv_i = it->feature_per_frame[0].point; // 取出该特征点在第一帧KF中的归一化坐标
      it->feature_per_frame.erase(it->feature_per_frame.begin()); // 由于该点不再被原来第一帧看到，因此删除该帧的观测
      if (it->feature_per_frame.size() < 2) // 如果该特征点只被一帧看到，那么删除该特征点
      {
        feature.erase(it);
        continue;
      }
      else
      {
        Eigen::Vector3d pts_i = uv_i * it->estimated_depth; // 实际坐标系下的坐标
        Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P; // 世界坐标系下的坐标
        Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P); // 新的最老帧相机坐标系下的坐标
        double dep_j = pts_j(2);
        if (dep_j > 0)
          it->estimated_depth = dep_j;
        else
          it->estimated_depth = INIT_DEPTH;
      }
    }
    // remove tracking-lost feature after marginalize
    /*
    if (it->endFrame() < WINDOW_SIZE - 1)
    {
        feature.erase(it);
    }
    */
  }
}

// 如果还没有初始化成功，因此不进行地图点新的深度的换算，因为此时还要进行视觉惯性对齐
void FeatureManager::removeBack()
{
  for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
  {
      it_next++;

      if (it->start_frame != 0)
        it->start_frame--;
      else
      {
        it->feature_per_frame.erase(it->feature_per_frame.begin());
        if (it->feature_per_frame.empty())
          feature.erase(it);
      }
  }
}

// 对margin倒数第二帧进行处理
void FeatureManager::removeFront(int frame_count)
{
  for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
  {
    it_next++;

    if (it->start_frame == frame_count) // 由于地图点被最后一帧看到，由于滑窗，它的起始帧id-1
    {
      it->start_frame--;
    }
    else
    {
      int j = WINDOW_SIZE - 1 - it->start_frame; // 倒数第二帧在这个地图点对应的KF vector中的位置
      if (it->endFrame() < frame_count - 1) // 如果只有一帧观测，那么没什么好做的
        continue;
      it->feature_per_frame.erase(it->feature_per_frame.begin() + j); // 可以被倒数第二帧看到，那么删除倒数第二帧的观测
      if (it->feature_per_frame.empty())
        feature.erase(it);
    }
  }
}

double FeatureManager::compensatedParallax2(const FeaturePerId& it_per_id, int frame_count)
{
  // check the second last frame is keyframe or not
  // parallax between second last frame and third last frame
  const FeaturePerFrame& frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
  const FeaturePerFrame& frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

  double ans;

  Vector3d p_j = frame_j.point; // 归一化坐标
  double u_j = p_j(0);
  double v_j = p_j(1);

  Vector3d p_i = frame_i.point;
  double u_i = p_i(0);
  double v_i = p_i(1);
  double du = u_i - u_j, dv = v_i - v_j; // 特征点在两帧图像上的归一化坐标差

  ans = sqrt(du * du + dv * dv);

  return ans;
}