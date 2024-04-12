#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

// 判断特征点是否在图像边界内（远离边界）
bool inBorder(const cv::Point2f& pt)
{
  const int BORDER_SIZE = 1;
  int img_x = cvRound(pt.x);
  int img_y = cvRound(pt.y);
  return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

// 去除status为0的特征点
void reduceVector(vector<cv::Point2f>& v, vector<uchar> status)
{
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i])
      v[j++] = v[i]; // 使用 v[i] 的值为 v[j] 赋值，然后 j 增加 1
  v.resize(j);
}

void reduceVector(vector<int>& v, vector<uchar> status)
{
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i])
      v[j++] = v[i];
  v.resize(j);
}


FeatureTracker::FeatureTracker() = default;

void FeatureTracker::setMask()
{
  if (FISHEYE)
    mask = fisheye_mask.clone();
  else
    mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255)); // 用scalar初始化矩阵


  // prefer to keep features that are tracked for long time
  // pair的第一个元素是特征点的跟踪次数，第二个元素是特征点的坐标和id
  vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;
  for (unsigned int i = 0; i < forw_pts.size(); i++)
    cnt_pts_id.emplace_back(track_cnt[i], make_pair(forw_pts[i], ids[i]));

  // 按照特征点的跟踪次数从大到小排序，使用lambda表达式定义了一个比较函数
  // []捕获列表可以捕获当前函数作用域的零个或多个变量
  sort(cnt_pts_id.begin(), cnt_pts_id.end(),
       [](const pair<int, pair<cv::Point2f, int>>& a, const pair<int, pair<cv::Point2f, int>>& b)
  {
    return a.first > b.first;
  });

  forw_pts.clear();
  ids.clear();
  track_cnt.clear();

  for (auto& it: cnt_pts_id)
  {
    if (mask.at<uchar>(it.second.first) == 255) // 这里不需要判断
    {
      forw_pts.push_back(it.second.first);
      ids.push_back(it.second.second);
      track_cnt.push_back(it.first);
      // 以特征点为圆心，半径为MIN_DIST的圆内的像素点都设置为0，thickness为正表示画圆的轮廓宽度，为负表示填充圆
      cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
    }
  }
}

// 添加新的特征点，ids初始化为-1，track_cnt初始化为1
void FeatureTracker::addPoints()
{
  for (auto& p: n_pts)
  {
    forw_pts.push_back(p); // 将新的特征点放到forw_pts（当前帧的特征点）中
    ids.push_back(-1);
    track_cnt.push_back(1);
  }
}

/**
 * @brief
 *
 * @param _img 输入图像
 * @param _cur_time 图像的时间戳
 * 1、图像均衡化预处理
 * 2、光流追踪
 * 3、提取新的特征点（如果发布）
 * 4、所以特征点去畸变，计算速度
 */
void FeatureTracker::readImage(const cv::Mat& _img, double _cur_time)
{
  cv::Mat img;
  TicToc t_r;
  cur_time = _cur_time;
  // clahe（限制对比度自适应直方图均衡化）是对原算法的改进，解决了： 1.全局性问题；2.背景噪声增强问题
  if (EQUALIZE)
  {
    // 图像过亮或者过暗，提取特征点比较困难，clahe可以提高图像的对比度，方便提取特征点
    // cv::CLAHE是实现clahe算法的类，clipLimit: 这是对局部直方图均衡化的对比度的限制，tileGridSize: 这定义了用于局部直方图均衡化的块的大小
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    TicToc t_c;
    // 对灰度图像进行clahe
    clahe->apply(_img, img);
    ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
  }
  else
    img = _img;

  // forw_img是当前帧图像，cur_img是上一帧图像
  if (forw_img.empty()) // prev_img没使用到
  {
    prev_img = cur_img = forw_img = img;
  }
  else
  {
    forw_img = img;
  }

  forw_pts.clear(); // 清空当前帧的特征点

  // 上一帧有特征点，才能进行光流追踪
  if (!cur_pts.empty())
  {
    TicToc t_o;
    vector<uchar> status; // 每一个特征点的状态，1表示成功跟踪到，0表示跟踪失败
    vector<float> err; // 每一个特征点的追踪误差
    // step1. 通过opencv光流追踪得到的状态位剔除outlier
    cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

    // 默认情况下，光流追踪的特征点数量和输入的特征点数量可以不一样
    for (int i = 0; i < int(forw_pts.size()); i++)
      // step2. 通过图像边界剔除outlier
      if (status[i] && !inBorder(forw_pts[i]))
        status[i] = 0;
    // 去除status为0的特征点
    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(ids, status); // 特征点的id
    reduceVector(cur_un_pts, status); // 去畸变后的特征点
    reduceVector(track_cnt, status); // 特征点的跟踪次数
    ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
  }

  // 被追踪到的特征点是之前就存在的，所以track_cnt++
  for (auto& n: track_cnt) // 循环只有一行时，可以不用大括号
    n++;

  // 按照目前的设置，隔一帧发布一次特征点
  if (PUB_THIS_FRAME)
  {
    // step3. 通过对极约束剔除outlier
    rejectWithF();
    ROS_DEBUG("set mask begins");
    TicToc t_m;
    // 给现有的特征点设置mask，避免在附近提取新的特征点
    setMask();
    ROS_DEBUG("set mask costs %fms", t_m.toc());

    ROS_DEBUG("detect feature begins");
    TicToc t_t;
    // MAX_CNT是允许的最大的特征点数量
    int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
    if (n_max_cnt > 0)
    {
      if (mask.empty())
        cout << "mask is empty " << endl;
      if (mask.type() != CV_8UC1)
        cout << "mask type wrong " << endl;
      if (mask.size() != forw_img.size())
        cout << "wrong size " << endl; // 追踪的特征点数目保持在150个左右
      cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - (int)forw_pts.size(), 0.01, MIN_DIST, mask); // 提取新的特征点，Harris角点检测
    }
    else
      n_pts.clear();
    ROS_DEBUG("detect feature costs: %fms", t_t.toc());

    ROS_DEBUG("add feature begins");
    TicToc t_a;
    addPoints();
    ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
  }
  prev_img = cur_img; // unused
  prev_pts = cur_pts; // unused
  prev_un_pts = cur_un_pts; // unused
  cur_img = forw_img; // 当前帧图像赋给上一帧图像
  cur_pts = forw_pts; // 当前帧特征点赋给上一帧特征点
  undistortedPoints();
  prev_time = cur_time;
}

void FeatureTracker::rejectWithF()
{
  // 当前被追踪的特征点数量大于8个，才能进行ransac
  if (forw_pts.size() >= 8)
  {
    ROS_DEBUG("FM ransac begins");
    TicToc t_f;
    vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
      Eigen::Vector3d tmp_p;
      // 得到归一化坐标系的值
      m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
      // 这里使用了一个虚拟相机，将归一化坐标系的值转换到虚拟相机的像素坐标系，好处是对F_THRESHOLD和相机无关（因为去完畸变以后的归一化坐标与相机模型无关）
      tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
      tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
      un_cur_pts[i] = cv::Point2f(float(tmp_p.x()), float(tmp_p.y())); // 去完畸变的虚拟相机的像素坐标

      m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
      tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
      tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
      un_forw_pts[i] = cv::Point2f(float(tmp_p.x()), float(tmp_p.y()));
    }

    vector<uchar> status;
    cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status); // 去除outlier
    int size_a = (int)cur_pts.size();
    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(cur_un_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
    ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
    ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
  }
}

// 给新的特征点分配id，如果超过了ids的大小，就返回false
bool FeatureTracker::updateID(unsigned int i)
{
  if (i < ids.size())
  {
    if (ids[i] == -1)
      ids[i] = n_id++; // 当前特征点的最大id
    return true;
  }
  else
    return false;
}

void FeatureTracker::readIntrinsicParameter(const string& calib_file)
{
  ROS_INFO("reading parameter of camera %s", calib_file.c_str());
  m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string& name)
{
  // 创建一个新的图像，大小为原图像加上600
  cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
  vector<Eigen::Vector2d> distortedp, undistortedp;
  for (int i = 0; i < COL; i++) {
    for (int j = 0; j < ROW; j++) {
      Eigen::Vector2d a(i, j);
      Eigen::Vector3d b;
      m_camera->liftProjective(a, b);
      distortedp.push_back(a);
      undistortedp.emplace_back(b.x() / b.z(), b.y() / b.z());
      //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
    }
  }

  for (int i = 0; i < int(undistortedp.size()); i++)
  {
    cv::Mat pp(3, 1, CV_32FC1);
    pp.at<float>(0, 0) = float(undistortedp[i].x() * FOCAL_LENGTH) + (float)COL / 2;
    pp.at<float>(1, 0) = float(undistortedp[i].y() * FOCAL_LENGTH) + (float)ROW / 2;
    pp.at<float>(2, 0) = 1.0;
    //cout << trackerData[0].K << endl;
    //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
    //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
    if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < (float)ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < (float)COL + 600)
    {
      undistortedImg.at<uchar>((int)pp.at<float>(1, 0) + 300, (int)pp.at<float>(0, 0) + 300) = cur_img.at<uchar>((int)distortedp[i].y(), (int)distortedp[i].x());
    }
    else
    {
        //ROS_ERROR("(%f %f) -> (%f %f)", distorted p[i].y, distorted p[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
    }
  }
  cv::imshow(name, undistortedImg);
  cv::waitKey(0);
}

// 当前帧所有特征点去畸变，计算速度，用来后续时间戳标定
void FeatureTracker::undistortedPoints()
{
  cur_un_pts.clear();
  cur_un_pts_map.clear();
  //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
  for (unsigned int i = 0; i < cur_pts.size(); i++)
  {
    // 有些点已经去过畸变了，这里连同新加入的点一起去畸变
    Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
    Eigen::Vector3d b;
    m_camera->liftProjective(a, b); // 去畸变
    cur_un_pts.emplace_back(b.x() / b.z(), b.y() / b.z());
    // id->去畸变后的特征点坐标
    cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(float(b.x() / b.z()), float(b.y() / b.z()))));
    //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
  }
  // calculate points velocity
  if (!prev_un_pts_map.empty())
  {
    double dt = cur_time - prev_time;
    pts_velocity.clear();
    for (unsigned int i = 0; i < cur_un_pts.size(); i++)
    {
      if (ids[i] != -1)
      {
        std::map<int, cv::Point2f>::iterator it;
        it = prev_un_pts_map.find(ids[i]);
        // 如果上一帧图像中存在该特征点，计算速度
        if (it != prev_un_pts_map.end())
        {
          double v_x = (cur_un_pts[i].x - it->second.x) / dt;
          double v_y = (cur_un_pts[i].y - it->second.y) / dt;
          pts_velocity.emplace_back(v_x, v_y);
        }
        else
          pts_velocity.emplace_back(0, 0);
      }
      else
      {
        pts_velocity.emplace_back(0, 0);
      }
    }
  }
  else
  {
    // 第一帧图像，速度为0
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
      pts_velocity.emplace_back(0, 0);
    }
  }
  prev_un_pts_map = cur_un_pts_map;
}
