#include "parameters.h"

std::string IMAGE_TOPIC;
std::string IMU_TOPIC;
std::vector<std::string> CAM_NAMES; // 存放相机配置文件的名字
std::string FISHEYE_MASK;
int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
int STEREO_TRACK;
int EQUALIZE;
int ROW;
int COL;
int FOCAL_LENGTH;
int FISHEYE;
bool PUB_THIS_FRAME;

template <typename T>
T readParam(ros::NodeHandle& n, std::string name)
{
  T ans;
  if (n.getParam(name, ans)) // 从参数服务器中读取参数，name为键值，ans为读取的值
  {
    ROS_INFO_STREAM("Loaded " << name << ": " << ans); // ROS_INFO是传统的格式化字符串风格的输出，ROS_INFO_STREAM是C++流式风格的输出
  }
  else
  {
    ROS_ERROR_STREAM("Failed to load " << name);
    n.shutdown();
  }
  return ans;
}

void readParameters(ros::NodeHandle& n)
{
  std::string config_file;
  // 获得配置文件的路径
  config_file = readParam<std::string>(n, "config_file");
  // 使用opencv的yaml文件接口读取配置文件
  cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
  if (!fsSettings.isOpened())
  {
    std::cerr << "ERROR: Wrong path to settings" << std::endl;
  }
  // 获取vins-mono源码存放的路径
  std::string VINS_FOLDER_PATH = readParam<std::string>(n, "vins_folder");
 
  fsSettings["image_topic"] >> IMAGE_TOPIC;
  fsSettings["imu_topic"] >> IMU_TOPIC;
  MAX_CNT = fsSettings["max_cnt"]; // 单帧图像中最多提取的特征点数
  MIN_DIST = fsSettings["min_dist"]; // 两个特征点之间的最小的像素距离
  ROW = fsSettings["image_height"];
  COL = fsSettings["image_width"];
  FREQ = fsSettings["freq"]; // 发布追踪结果的频率，原图像的频率为20Hz
  F_THRESHOLD = fsSettings["F_threshold"]; // 对极约束ransac求解的inlier阈值
  SHOW_TRACK = fsSettings["show_track"]; // 是否显示追踪结果
  EQUALIZE = fsSettings["equalize"]; // 是否进行直方图均衡化
  FISHEYE = fsSettings["fisheye"]; // 是否是鱼眼相机
  if (FISHEYE == 1)
    FISHEYE_MASK = VINS_FOLDER_PATH + "config/fisheye_mask.jpg";
  CAM_NAMES.push_back(config_file);

  WINDOW_SIZE = 20; // 光流的窗口大小
  STEREO_TRACK = false;
  FOCAL_LENGTH = 460; // 相机的焦距
  PUB_THIS_FRAME = false; // 是否发布这一帧图像

  if (FREQ == 0)
    FREQ = 100;

  fsSettings.release();

}
