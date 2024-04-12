#include <cstdio>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf; // imu数据的共享队列
queue<sensor_msgs::PointCloudConstPtr> feature_buf; // 图像数据的共享队列
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;

std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;
Eigen::Vector3d tmp_P{0, 0, 0};
Eigen::Quaterniond tmp_Q{1, 0, 0, 0}; // Eigen的默认构造，值是未初始化的（随机的，与之前内存里的值有关）
Eigen::Vector3d tmp_V{0, 0, 0};
Eigen::Vector3d tmp_Ba{0, 0, 0};
Eigen::Vector3d tmp_Bg{0, 0, 0};
Eigen::Vector3d acc_0{0, 0, 0};
Eigen::Vector3d gyr_0{0, 0, 0};
bool init_feature = false;
bool init_imu = true;
double last_imu_t = 0;
ofstream ofs("/home/zhang/vins_estimator.txt", ios::app);
// 根据imu数据预测位姿，系统未初始化前，tmp_P, tmp_Q, tmp_V, tmp_Ba, tmp_Bg值都是不对的，他们的作用是在初始化完成以后，发布最新的里程计结果
void predict(const sensor_msgs::ImuConstPtr& imu_msg)
{
  double t = imu_msg->header.stamp.toSec();
  if (init_imu) // 判断是不是第一帧
  {
    latest_time = t;
    init_imu = false;
    return;
  }
  
  double dt = t - latest_time; // imu测量的时间间隔
  latest_time = t;

  // imu数据的线性加速度和角速度
  double dx = imu_msg->linear_acceleration.x;
  double dy = imu_msg->linear_acceleration.y;
  double dz = imu_msg->linear_acceleration.z;
  Eigen::Vector3d linear_acceleration{dx, dy, dz};

  double rx = imu_msg->angular_velocity.x;
  double ry = imu_msg->angular_velocity.y;
  double rz = imu_msg->angular_velocity.z;
  Eigen::Vector3d angular_velocity{rx, ry, rz};

  // 上一时刻的真实加速度值
  Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g; // 上一时刻的世界坐标系下的加速度值，这里的tmp_Q是上一时刻的旋转矩阵Rwi，w->i

  // 陀螺仪的中值结果
  Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
  // 更新姿态
  tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

  Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

  Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

  // 更新位置和速度
  tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
  tmp_V = tmp_V + dt * un_acc;

  acc_0 = linear_acceleration;
  gyr_0 = angular_velocity;
}

// 用最新VIO结果更新imu对应的位姿
void update()
{
  TicToc t_predict;
  latest_time = current_time;
  tmp_P = estimator.Ps[WINDOW_SIZE];
  tmp_Q = estimator.Rs[WINDOW_SIZE];
  tmp_V = estimator.Vs[WINDOW_SIZE];
  tmp_Ba = estimator.Bas[WINDOW_SIZE];
  tmp_Bg = estimator.Bgs[WINDOW_SIZE];
  acc_0 = estimator.acc_0;
  gyr_0 = estimator.gyr_0;

  queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf; // 拷贝一份imu数据，因为下面需要pop
  for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
    predict(tmp_imu_buf.front()); // 得到最新的imu时刻对应的位姿
}

// 做图像数据和imu数据的同步
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
  std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

  while (true)
  {
    if (imu_buf.empty() || feature_buf.empty())
      return measurements;

    // imu数据的最新时间戳小于图像数据的最老时间戳，说明imu数据比较旧
    if (imu_buf.back()->header.stamp.toSec() <= feature_buf.front()->header.stamp.toSec() + estimator.td) // T_imu = T_cam + td
    {
      // ROS_WARN("wait for imu, only should happen at the beginning");
      sum_of_wait++;
      return measurements;
    }

    // imu数据的最老时间戳大于图像数据的最老时间戳，需要丢弃一些图像数据
    if (imu_buf.front()->header.stamp.toSec() >= feature_buf.front()->header.stamp.toSec() + estimator.td)
    {
      ROS_WARN("throw img, only should happen at the beginning");
      feature_buf.pop();
      continue;
    }

    // 此时就可以保证最老的图像数据前一定有imu数据
    sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front(); // 拿出最老的图像数据
    feature_buf.pop();

    std::vector<sensor_msgs::ImuConstPtr> IMUs;
    // 一般第一帧不会严格对齐，但是后面就会对齐，当然第一帧也不会用到
    while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
    {
      IMUs.emplace_back(imu_buf.front()); // 将最老的图像数据前的imu数据加入到IMUs中
      imu_buf.pop();
    }
    // 保留图像时间戳后一个imu数据，但不会从imu_buf中删除
    IMUs.emplace_back(imu_buf.front());
    if (IMUs.empty())
      ROS_WARN("no imu between two image");
    measurements.emplace_back(IMUs, img_msg);
    return measurements;
  }
}

/**
 * @brief imu消息存进buffer，同时按照imu频率预测位姿并发布，这样可以提高里程计的频率
 * @param imu_msg
 */
void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg)
{
  if (imu_msg->header.stamp.toSec() <= last_imu_t)
  {
    ROS_WARN("imu message in disorder!");
    return;
  }
  last_imu_t = imu_msg->header.stamp.toSec(); // 将最新的imu时间戳赋值给last_imu_t

  // 消费者生产者模型，一个线程往队列中放入数据，一个线程往队列中取数据，取数据前需要判断一下队列中确实有数据，由于这个队列是线程间共享的，所以，需要使用互斥锁进行保护，
  // 一个线程在往队列添加数据的时候，另一个线程不能取，反之亦然
  m_buf.lock(); // 加锁
  imu_buf.push(imu_msg); // 将新来的imu数据加入到一个队列中
  m_buf.unlock();
  con.notify_one(); // 通知处理线程队列中有数据，条件变量有两个重要函数wait和notify_one，wait可以让线程陷入休眠状态，而notify_one则可以唤醒休眠的线程

  {
    std::lock_guard<std::mutex> lg(m_state); // 范围保护锁，离开自动解锁
    predict(imu_msg);
    std_msgs::Header header = imu_msg->header;
    header.frame_id = "world";
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) // 初始化已经完成
      pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header); // 发布最新的里程计结果
  }
}

/**
 * @brief 特征点消息存进buffer
 * @param feature_msg
 */
void feature_callback(const sensor_msgs::PointCloudConstPtr& feature_msg)
{
  if (!init_feature)
  {
    //skip the first detected feature, which doesn't contain optical flow speed
    init_feature = true;
    return;
  }
  m_buf.lock();
  feature_buf.push(feature_msg);
  m_buf.unlock();
  con.notify_one();
}

/**
 * @brief 重启估计器
 * @param restart_msg
 */
void restart_callback(const std_msgs::BoolConstPtr& restart_msg)
{
  if (restart_msg->data == true)
  {
    ROS_WARN("restart the estimator!");
    m_buf.lock();
    while(!feature_buf.empty())
      feature_buf.pop();
    while(!imu_buf.empty())
      imu_buf.pop();
    m_buf.unlock();
    m_estimator.lock();
    estimator.clearState();
    estimator.setParameter();
    m_estimator.unlock();
    current_time = -1;
    last_imu_t = 0;
  }
}

void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
  //printf("relocalization callback! \n");
  m_buf.lock();
  relo_buf.push(points_msg);
  m_buf.unlock();
}

// thread: visual-inertial odometry
void process()
{
  while (true) // 一直循环，直到程序结束
  {
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements; // imu和图像数据
    std::unique_lock<std::mutex> lk(m_buf);
    /*
     * 在管理互斥锁的时候，使用的是std::unique_lock而不是std::lock_guard，而且事实上也不能使用std::lock_guard。这需要先解释下wait()函数所做的事情。
     * 可以看到，在wait()函数之前，使用互斥锁保护了，如果wait的时候什么都没做，岂不是一直持有互斥锁？那生产者也会一直卡住，不能够将数据放入队列中了。
     * 所以，wait()函数会先调用互斥锁的unlock()函数，然后再将自己睡眠，在被唤醒后，又会继续持有锁，保护后面的队列操作。
     * 而lock_guard没有lock和unlock接口，而unique_lock提供了。这就是必须使用unique_lock的原因
     * 可以将cond.wait(locker)函数换一种写法，wait()的第二个参数可以传入一个函数表示检查条件，这里使用lambda函数最为简单，如果这个函数返回的是true，
     * wait()函数不会阻塞会直接返回，如果这个函数返回的是false，wait()函数就会阻塞着等待唤醒，如果被伪唤醒，会继续判断函数返回值
     */
    con.wait(lk, [&]
    {
      return !(measurements = getMeasurements()).empty();
    });
    lk.unlock();
    m_estimator.lock(); 
    
    // 进行后端求解，数据集可以做到imu和图像时间戳严格对齐，每一次获取的measurements中，
    // 包含有一个图像数据和小于它的多个imu数据，以及等于或大于（取决于是否严格对齐）它一个imu数据
    for (auto& measurement: measurements)
    {
      auto img_msg = measurement.second;
      double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
      for (auto& imu_msg: measurement.first) // 遍历imu数据，老的inu数据在前，新的imu数据在后
      {
        double t = imu_msg->header.stamp.toSec();
        double img_t = img_msg->header.stamp.toSec() + estimator.td; // T_cam' = T_imu = T_cam + td
        if (t <= img_t) // 最后一个之前的imu数据
        {
          if (current_time < 0)
            current_time = t; // current_time是前一帧imu数据的时间戳
          double dt = t - current_time;
          ROS_ASSERT(dt >= 0);
          current_time = t;
          dx = imu_msg->linear_acceleration.x;
          dy = imu_msg->linear_acceleration.y;
          dz = imu_msg->linear_acceleration.z;
          rx = imu_msg->angular_velocity.x;
          ry = imu_msg->angular_velocity.y;
          rz = imu_msg->angular_velocity.z;
          estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz)); // 第1个imu数据时，dt=0
        }
        else // 针对最后一个imu数据，进行线性插值
        {
          double dt_1 = img_t - current_time;
          double dt_2 = t - img_t;
          current_time = img_t;
          ROS_ASSERT(dt_1 >= 0);
          ROS_ASSERT(dt_2 >= 0);
          ROS_ASSERT(dt_1 + dt_2 > 0);
          double w1 = dt_2 / (dt_1 + dt_2);
          double w2 = dt_1 / (dt_1 + dt_2);
          dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
          dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
          dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
          rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
          ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
          rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
          estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
        }
      }
      
      // set relocalization frame
      // 回环相关部分
      sensor_msgs::PointCloudConstPtr relo_msg = nullptr;
      while (!relo_buf.empty()) // 从队列中取出最新回环数据
      {
        relo_msg = relo_buf.front();
        relo_buf.pop();
      }
      if (relo_msg != nullptr) // 有效回环数据
      {
        vector<Vector3d> match_points;
        double frame_stamp = relo_msg->header.stamp.toSec(); // 回环帧的时间戳
        for (auto point: relo_msg->points)
        {
          Vector3d u_v_id;
          u_v_id.x() = point.x;
          u_v_id.y() = point.y;
          u_v_id.z() = point.z; // 回环帧的归一化坐标和特征点id
          match_points.push_back(u_v_id);
        }
        // 回环帧的位姿
        Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
        Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
        Matrix3d relo_r = relo_q.toRotationMatrix();
        int frame_index;
        frame_index = (int)relo_msg->channels[0].values[7];
        estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
      }

      ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

      TicToc t_s;
      // 特征点id->特征点数据
      map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
      for (unsigned int i = 0; i < img_msg->points.size(); i++)
      {
        int v = (int)img_msg->channels[0].values[i]; // 第一个通道是特征点id
        int feature_id = v / NUM_OF_CAM;
        int camera_id = v % NUM_OF_CAM;
        double x = img_msg->points[i].x; // 归一化坐标
        double y = img_msg->points[i].y;
        double z = img_msg->points[i].z;
        double p_u = img_msg->channels[1].values[i];
        double p_v = img_msg->channels[2].values[i];
        double velocity_x = img_msg->channels[3].values[i];
        double velocity_y = img_msg->channels[4].values[i];
        ROS_ASSERT(z == 1); // 检查是否是归一化坐标
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
      }
      estimator.processImage(image, img_msg->header);

      double whole_t = t_s.toc();
      printStatistics(estimator, whole_t);
      std_msgs::Header header = img_msg->header;
      header.frame_id = "world";

      pubOdometry(estimator, header);
      pubKeyPoses(estimator, header);
      pubCameraPose(estimator, header);
      pubPointCloud(estimator, header);
      pubTF(estimator, header);
      pubKeyframe(estimator);
      if (relo_msg != nullptr)
        pubRelocalization(estimator);
      //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
    }
    m_estimator.unlock();
    m_buf.lock();
    m_state.lock();
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
      update();
    m_state.unlock();
    m_buf.unlock();
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "vins_estimator");
  ros::NodeHandle n("~");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
  readParameters(n);
  estimator.setParameter();
#ifdef EIGEN_DONT_PARALLELIZE
  ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
  ROS_WARN("waiting for image and imu...");

  // 注册一些Publisher
  registerPub(n);
  
  // 订阅一些消息，在ROS，通信方式的选择和优化对于实时性或性能要求较高的应用至关重要。ros::TransportHints 是一个ROS中用于配置话题传输的工具
  // tcpNoDelay() 时，你实际上是在请求使用TCP_NODELAY选项，这会禁用Nagle算法。结果是，消息会更快地发送出去，而不是等待小数据包被聚合为更大的数据包。
  // 这特别适用于需要快速响应的应用，例如机器人的遥控。Nagle算法是设计来优化小数据包在TCP中的传输的，但对于需要低延迟的应用来说，这可能是一个问题
  ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
  ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
  ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
  // 回环检测的fast relocalization响应
  ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

  // 核心处理线程，创建一个线程来执行process函数
  std::thread measurement_process{process};
  ros::spin(); // 当你调用 ros::spin()，你启动了一个简单的单线程模式，所有来自不同订阅者的回调都会在这个单一线程中顺序执行

  return 0;
}
