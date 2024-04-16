#include <vector>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <ros/package.h>
#include <mutex>
#include <queue>
#include <thread>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "keyframe.h"
#include "utility/tic_toc.h"
#include "pose_graph.h"
#include "utility/CameraPoseVisualization.h"
#include "parameters.h"
#define SKIP_FIRST_CNT 10
using namespace std;

queue<sensor_msgs::ImageConstPtr> image_buf;
queue<sensor_msgs::PointCloudConstPtr> point_buf;
queue<nav_msgs::Odometry::ConstPtr> pose_buf;
queue<Eigen::Vector3d> odometry_buf;
std::mutex m_buf;
std::mutex m_process;
int frame_index  = 0;
int sequence = 1;
PoseGraph posegraph;
int skip_first_cnt = 0;
int SKIP_CNT;
int skip_cnt = 0;
bool load_flag = false;
bool start_flag = false;
double SKIP_DIS = 0;

int VISUALIZATION_SHIFT_X;
int VISUALIZATION_SHIFT_Y;
int ROW;
int COL;
int DEBUG_IMAGE;
int VISUALIZE_IMU_FORWARD;
int LOOP_CLOSURE;
int FAST_RELOCALIZATION;

camodocal::CameraPtr m_camera;
Eigen::Vector3d tic;
Eigen::Matrix3d qic;
ros::Publisher pub_match_img;
ros::Publisher pub_match_points;
ros::Publisher pub_camera_pose_visual;
ros::Publisher pub_key_odometrys;
ros::Publisher pub_vio_path;
nav_msgs::Path no_loop_path;

std::string BRIEF_PATTERN_FILE;
std::string POSE_GRAPH_SAVE_PATH;
std::string VINS_RESULT_PATH;
CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
Eigen::Vector3d last_t(-100, -100, -100);
double last_image_time = -1;

void new_sequence()
{
  printf("new sequence\n");
  sequence++;
  printf("sequence cnt %d \n", sequence);
  if (sequence > 5)
  {
    ROS_WARN("only support 5 sequences since it's boring to copy code for more sequences.");
    ROS_BREAK();
  }
  posegraph.posegraph_visualization->reset();
  posegraph.publish();
  m_buf.lock();
  while(!image_buf.empty())
    image_buf.pop();
  while(!point_buf.empty())
    point_buf.pop();
  while(!pose_buf.empty())
    pose_buf.pop();
  while(!odometry_buf.empty())
    odometry_buf.pop();
  m_buf.unlock();
}

/**
 * @brief 接受原图像消息
 * @param image_msg
 */
void image_callback(const sensor_msgs::ImageConstPtr& image_msg)
{
  if (!LOOP_CLOSURE)
    return;
  m_buf.lock();
  image_buf.push(image_msg); // 上锁，并将图像消息放入队列
  m_buf.unlock();

  // detect unstable camera stream
  if (last_image_time == -1)
    last_image_time = image_msg->header.stamp.toSec();
  // 检查是否存在时间戳错乱或者延时过大的情况
  else if (image_msg->header.stamp.toSec() - last_image_time > 1.0 || image_msg->header.stamp.toSec() < last_image_time)
  {
    ROS_WARN("image discontinue! detect a new sequence!");
    new_sequence(); // 创建一个新的sequence，相当于地图
  }
  last_image_time = image_msg->header.stamp.toSec();
}

void point_callback(const sensor_msgs::PointCloudConstPtr& point_msg)
{
  if(!LOOP_CLOSURE)
    return;
  m_buf.lock();
  point_buf.push(point_msg);
  m_buf.unlock();
}

void pose_callback(const nav_msgs::Odometry::ConstPtr& pose_msg)
{
  if (!LOOP_CLOSURE)
    return;
  m_buf.lock();
  pose_buf.push(pose_msg);
  m_buf.unlock();
}

/**
 * @brief 发布经过回环修正后的最新位姿
 * @param forward_msg
 */
void imu_forward_callback(const nav_msgs::Odometry::ConstPtr& forward_msg)
{
  if (VISUALIZE_IMU_FORWARD)
  {
    Vector3d vio_t(forward_msg->pose.pose.position.x, forward_msg->pose.pose.position.y, forward_msg->pose.pose.position.z);
    Quaterniond vio_q;
    vio_q.w() = forward_msg->pose.pose.orientation.w;
    vio_q.x() = forward_msg->pose.pose.orientation.x;
    vio_q.y() = forward_msg->pose.pose.orientation.y;
    vio_q.z() = forward_msg->pose.pose.orientation.z;

    vio_t = posegraph.w_r_vio * vio_t + posegraph.w_t_vio;
    vio_q = posegraph.w_r_vio *  vio_q;

    vio_t = posegraph.r_drift * vio_t + posegraph.t_drift;
    vio_q = posegraph.r_drift * vio_q;

    Vector3d vio_t_cam;
    Quaterniond vio_q_cam;
    vio_t_cam = vio_t + vio_q * tic;
    vio_q_cam = vio_q * qic;

    cameraposevisual.reset();
    cameraposevisual.add_pose(vio_t_cam, vio_q_cam);
    cameraposevisual.publish_by(pub_camera_pose_visual, forward_msg->header);
  }
}

// 利用VIO进行重定位的结果进行修正
void relo_relative_pose_callback(const nav_msgs::Odometry::ConstPtr& pose_msg)
{
  Vector3d relative_t = Vector3d(pose_msg->pose.pose.position.x,
                                 pose_msg->pose.pose.position.y,
                                 pose_msg->pose.pose.position.z);
  Quaterniond relative_q;
  relative_q.w() = pose_msg->pose.pose.orientation.w;
  relative_q.x() = pose_msg->pose.pose.orientation.x;
  relative_q.y() = pose_msg->pose.pose.orientation.y;
  relative_q.z() = pose_msg->pose.pose.orientation.z;
  double relative_yaw = pose_msg->twist.twist.linear.x;
  int index = (int)pose_msg->twist.twist.linear.y; // 当前帧的id
  
  Eigen::Matrix<double, 8, 1> loop_info;
  loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
               relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
               relative_yaw;
  posegraph.updateKeyFrameLoop(index, loop_info);

}

/**
 * @brief 接受VIO中最新的位姿，不一定是KF（只要倒数第二帧以前的才是KF）
 * @param pose_msg
 */
void vio_callback(const nav_msgs::Odometry::ConstPtr& pose_msg)
{
  Vector3d vio_t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
  Quaterniond vio_q;
  vio_q.w() = pose_msg->pose.pose.orientation.w;
  vio_q.x() = pose_msg->pose.pose.orientation.x;
  vio_q.y() = pose_msg->pose.pose.orientation.y;
  vio_q.z() = pose_msg->pose.pose.orientation.z;

  vio_t = posegraph.w_r_vio * vio_t + posegraph.w_t_vio;
  vio_q = posegraph.w_r_vio *  vio_q;

  vio_t = posegraph.r_drift * vio_t + posegraph.t_drift;
  vio_q = posegraph.r_drift * vio_q;

  Vector3d vio_t_cam;
  Quaterniond vio_q_cam;
  vio_t_cam = vio_t + vio_q * tic;
  vio_q_cam = vio_q * qic;

  if (!VISUALIZE_IMU_FORWARD)
  {
    cameraposevisual.reset();
    cameraposevisual.add_pose(vio_t_cam, vio_q_cam);
    cameraposevisual.publish_by(pub_camera_pose_visual, pose_msg->header);
  }

  odometry_buf.push(vio_t_cam);
  if (odometry_buf.size() > 10)
  {
    odometry_buf.pop();
  }

  visualization_msgs::Marker key_odometrys;
  key_odometrys.header = pose_msg->header;
  key_odometrys.header.frame_id = "world";
  key_odometrys.ns = "key_odometrys";
  key_odometrys.type = visualization_msgs::Marker::SPHERE_LIST;
  key_odometrys.action = visualization_msgs::Marker::ADD;
  key_odometrys.pose.orientation.w = 1.0;
  key_odometrys.lifetime = ros::Duration();

  // static int key_odometrys_id = 0;
  key_odometrys.id = 0; // key_odometrys_id++;
  key_odometrys.scale.x = 0.1;
  key_odometrys.scale.y = 0.1;
  key_odometrys.scale.z = 0.1;
  key_odometrys.color.r = 1.0;
  key_odometrys.color.a = 1.0;

  for (unsigned int i = 0; i < odometry_buf.size(); i++)
  {
    geometry_msgs::Point pose_marker;
    Vector3d vio_t;
    vio_t = odometry_buf.front();
    odometry_buf.pop();
    pose_marker.x = vio_t.x();
    pose_marker.y = vio_t.y();
    pose_marker.z = vio_t.z();
    key_odometrys.points.push_back(pose_marker);
    odometry_buf.push(vio_t);
  }
  pub_key_odometrys.publish(key_odometrys);

  if (!LOOP_CLOSURE)
  {
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = pose_msg->header;
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose.position.x = vio_t.x();
    pose_stamped.pose.position.y = vio_t.y();
    pose_stamped.pose.position.z = vio_t.z();
    no_loop_path.header = pose_msg->header;
    no_loop_path.header.frame_id = "world";
    no_loop_path.poses.push_back(pose_stamped);
    pub_vio_path.publish(no_loop_path);
  }
}

void extrinsic_callback(const nav_msgs::Odometry::ConstPtr& pose_msg)
{
  m_process.lock();
  tic = Vector3d(pose_msg->pose.pose.position.x,
                 pose_msg->pose.pose.position.y,
                 pose_msg->pose.pose.position.z);
  qic = Quaterniond(pose_msg->pose.pose.orientation.w,
                    pose_msg->pose.pose.orientation.x,
                    pose_msg->pose.pose.orientation.y,
                    pose_msg->pose.pose.orientation.z).toRotationMatrix();
  m_process.unlock();
}

// 回环检测主要的处理函数
void process()
{
  if (!LOOP_CLOSURE)
    return;
  while (true)
  {
    sensor_msgs::ImageConstPtr image_msg = nullptr;
    sensor_msgs::PointCloudConstPtr point_msg = nullptr;
    nav_msgs::Odometry::ConstPtr pose_msg = nullptr;

    // find out the messages with same time stamp
    m_buf.lock();
    // 做一个时间戳的同步，涉及到原图，KF位姿以及KF对应的地图点
    if (!image_buf.empty() && !point_buf.empty() && !pose_buf.empty())
    { // 如果原图的时间戳比KF位姿的时间戳大，那么就抛弃KF位姿
      if (image_buf.front()->header.stamp.toSec() > pose_buf.front()->header.stamp.toSec())
      {
        pose_buf.pop();
        printf("throw pose at beginning\n");
      }
      else if (image_buf.front()->header.stamp.toSec() > point_buf.front()->header.stamp.toSec())
      {
        point_buf.pop();
        printf("throw point at beginning\n");
      }

      // 下面根据pose时间戳找时间戳同步的原图和地图点
      else if (image_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec()
               && point_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec())
      {
        pose_msg = pose_buf.front(); // 取出KF位姿
        pose_buf.pop();
        while (!pose_buf.empty()) // 清空KF位姿队列，回环的帧率慢一些没有关系
          pose_buf.pop();

        // 找到对应的pose原图
        while (image_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
          image_buf.pop();
        image_msg = image_buf.front();
        image_buf.pop();

        // 找到对应的pose地图点
        while (point_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
          point_buf.pop();
        point_msg = point_buf.front();
        point_buf.pop();
      }
    }
    m_buf.unlock();

    // 至此，找到了时间戳同步的KF位姿，KF原图和KF地图点，下面开始回环检测
    if (pose_msg != nullptr)
    {
      if (skip_first_cnt < SKIP_FIRST_CNT) // 跳过前SKIP_FIRST_CNT帧
      {
        skip_first_cnt++;
        continue;
      }

      if (skip_cnt < SKIP_CNT) // 降频，跳过前SKIP_CNT帧
      {
        skip_cnt++;
        continue;
      }
      else
      {
        skip_cnt = 0;
      }

      // 将ros消息转换为opencv格式
      cv_bridge::CvImageConstPtr ptr;
      if (image_msg->encoding == "8UC1")
      {
        sensor_msgs::Image img;
        img.header = image_msg->header;
        img.height = image_msg->height;
        img.width = image_msg->width;
        img.is_bigendian = image_msg->is_bigendian;
        img.step = image_msg->step;
        img.data = image_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
      }
      else
        ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);

      cv::Mat image = ptr->image;

      // build keyframe，得到KF位姿，转成eigen格式
      Vector3d T = Vector3d(pose_msg->pose.pose.position.x,
                            pose_msg->pose.pose.position.y,
                            pose_msg->pose.pose.position.z);
      Matrix3d R = Quaterniond(pose_msg->pose.pose.orientation.w,
                               pose_msg->pose.pose.orientation.x,
                               pose_msg->pose.pose.orientation.y,
                               pose_msg->pose.pose.orientation.z).toRotationMatrix();
      
      if ((T - last_t).norm() > SKIP_DIS) // 两帧平移距离大于SKIP_DIS才加入回环检测，SKIP_DIS=0
      {
        vector<cv::Point3f> point_3d; // VIO世界坐标系下的地图点坐标
        vector<cv::Point2f> point_2d_uv; // 像素坐标
        vector<cv::Point2f> point_2d_normal; // 归一化相机坐标系的坐标
        vector<double> point_id; // 地图点id

        for (unsigned int i = 0; i < point_msg->points.size(); i++)
        {
          cv::Point3f p_3d;
          p_3d.x = point_msg->points[i].x;
          p_3d.y = point_msg->points[i].y;
          p_3d.z = point_msg->points[i].z;
          point_3d.push_back(p_3d);

          cv::Point2f p_2d_uv, p_2d_normal;
          double p_id;
          p_2d_normal.x = point_msg->channels[i].values[0];
          p_2d_normal.y = point_msg->channels[i].values[1];
          p_2d_uv.x = point_msg->channels[i].values[2];
          p_2d_uv.y = point_msg->channels[i].values[3];
          p_id = point_msg->channels[i].values[4];
          point_2d_normal.push_back(p_2d_normal);
          point_2d_uv.push_back(p_2d_uv);
          point_id.push_back(p_id);
        }

        // 创建回环检测的KF
        auto* keyframe = new KeyFrame(pose_msg->header.stamp.toSec(), frame_index, T, R, image,
                                      point_3d, point_2d_uv, point_2d_normal, point_id, sequence);

        m_process.lock();
        start_flag = true;
        posegraph.addKeyFrame(keyframe, true); // 回环检测核心入口函数
        m_process.unlock();
        frame_index++;
        last_t = T;
      }
    }

    std::chrono::milliseconds dura(5);
    std::this_thread::sleep_for(dura);
  }
}

void command()
{
  if (!LOOP_CLOSURE)
    return;
  while (true)
  {
    char c = (char)getchar(); // 从标准输入读取一个字符
    if (c == 's') // s是保存地图
    {
      m_process.lock();
      posegraph.savePoseGraph();
      m_process.unlock();
      printf("save pose graph finish\nyou can set 'load_previous_pose_graph' to 1 in the config file to reuse it next time\n");
    }
    if (c == 'n') // n是新建一个sequence（一个新的地图）
      new_sequence();

    std::chrono::milliseconds dura(5);
    std::this_thread::sleep_for(dura);
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "pose_graph");
  ros::NodeHandle n("~");
  posegraph.registerPub(n);

  // read param
  n.getParam("visualization_shift_x", VISUALIZATION_SHIFT_X); // 这两个shift一般为0
  n.getParam("visualization_shift_y", VISUALIZATION_SHIFT_Y);
  n.getParam("skip_cnt", SKIP_CNT); // 跳过前SKIP_CNT帧
  n.getParam("skip_dis", SKIP_DIS); // 两帧距离的阈值
  std::string config_file;
  n.getParam("config_file", config_file);
  cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
  if(!fsSettings.isOpened())
  {
    std::cerr << "ERROR: Wrong path to settings" << std::endl;
  }

  double camera_visual_size = fsSettings["visualize_camera_size"];
  cameraposevisual.setScale(camera_visual_size);
  cameraposevisual.setLineWidth(camera_visual_size / 10.0);


  LOOP_CLOSURE = fsSettings["loop_closure"]; // 是否开启回环检测
  std::string IMAGE_TOPIC;
  int LOAD_PREVIOUS_POSE_GRAPH;
  if (LOOP_CLOSURE)
  {
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    std::string pkg_path = ros::package::getPath("pose_graph"); // 获取包路径
    string vocabulary_file = pkg_path + "/../support_files/brief_k10L6.bin"; // 训练好的二进制词袋路径
    cout << "vocabulary_file" << vocabulary_file << endl;
    posegraph.loadVocabulary(vocabulary_file); // 加载词袋

    BRIEF_PATTERN_FILE = pkg_path + "/../support_files/brief_pattern.yml"; // BRIEF描述子的点对选取模式
    cout << "BRIEF_PATTERN_FILE" << BRIEF_PATTERN_FILE << endl;
    m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(config_file);

    fsSettings["image_topic"] >> IMAGE_TOPIC;
    fsSettings["pose_graph_save_path"] >> POSE_GRAPH_SAVE_PATH;
    fsSettings["output_path"] >> VINS_RESULT_PATH;
    fsSettings["save_image"] >> DEBUG_IMAGE;

    // create folder if not exists
    FileSystemHelper::createDirectoryIfNotExists(POSE_GRAPH_SAVE_PATH.c_str());
    FileSystemHelper::createDirectoryIfNotExists(VINS_RESULT_PATH.c_str());

    VISUALIZE_IMU_FORWARD = fsSettings["visualize_imu_forward"]; // 可视化是否使用imu进行前推，默认为0
    LOAD_PREVIOUS_POSE_GRAPH = fsSettings["load_previous_pose_graph"]; // 是否加载之前的pose graph，默认为0
    FAST_RELOCALIZATION = fsSettings["fast_relocalization"]; // 是否开启快速重定位，和VIO节点有交互，默认为0
    VINS_RESULT_PATH = VINS_RESULT_PATH + "/vins_result_loop.csv";
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();
    fsSettings.release();

    if (LOAD_PREVIOUS_POSE_GRAPH)
    {
      printf("load pose graph\n");
      m_process.lock();
      posegraph.loadPoseGraph();
      m_process.unlock();
      printf("load pose graph finish\n");
      load_flag = true;
    }
    else
    {
      printf("no previous pose graph!\n");
      load_flag = true;
    }
  }

  fsSettings.release();

  ros::Subscriber sub_imu_forward = n.subscribe("/vins_estimator/imu_propagate", 2000, imu_forward_callback);
  ros::Subscriber sub_vio = n.subscribe("/vins_estimator/odometry", 2000, vio_callback);
  ros::Subscriber sub_image = n.subscribe(IMAGE_TOPIC, 2000, image_callback);
  ros::Subscriber sub_pose = n.subscribe("/vins_estimator/keyframe_pose", 2000, pose_callback); // 处理KF位姿
  ros::Subscriber sub_extrinsic = n.subscribe("/vins_estimator/extrinsic", 2000, extrinsic_callback);
  ros::Subscriber sub_point = n.subscribe("/vins_estimator/keyframe_point", 2000, point_callback);
  ros::Subscriber sub_relo_relative_pose = n.subscribe("/vins_estimator/relo_relative_pose", 2000, relo_relative_pose_callback);

  pub_match_img = n.advertise<sensor_msgs::Image>("match_image", 1000);
  pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 1000);
  pub_key_odometrys = n.advertise<visualization_msgs::Marker>("key_odometrys", 1000);
  pub_vio_path = n.advertise<nav_msgs::Path>("no_loop_path", 1000);
  pub_match_points = n.advertise<sensor_msgs::PointCloud>("match_points", 100);

  std::thread measurement_process;
  std::thread keyboard_command_process; // 键盘回调线程

  measurement_process = std::thread(process);
  keyboard_command_process = std::thread(command);

  ros::spin();

  return 0;
}
