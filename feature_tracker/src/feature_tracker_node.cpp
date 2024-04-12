#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img,pub_match;
ros::Publisher pub_restart;

// trackerData是一个FeatureTracker类型的数组，数组的长度是NUM_OF_CAM，即相机的个数
FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time; // 第一帧图像的时间戳
int pub_count = 1; // 如果当前帧发布了，发布的图像帧数
bool first_image_flag = true; // 是否是第一帧图像
double last_image_time = 0; // 上一帧图像的时间戳
bool init_pub = false;

void img_callback(const sensor_msgs::ImageConstPtr& img_msg)
{
  if (first_image_flag) // 判断是否是第一帧图像
  {
    first_image_flag = false;
    first_image_time = img_msg->header.stamp.toSec();
    last_image_time = img_msg->header.stamp.toSec();
    return;
  }
  // detect unstable camera stream光流追踪作用于连续的图像帧，如果图像帧之间的时间间隔大于1s，则认为图像流不稳定，或者出现时间戳错乱，那么就需要重置光流追踪器
  if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
  {
    ROS_WARN("image discontinue! reset the feature tracker!");
    first_image_flag = true;
    last_image_time = 0;
    pub_count = 1;
    std_msgs::Bool restart_flag;
    restart_flag.data = true;
    pub_restart.publish(restart_flag);
    return;
  }
  last_image_time = img_msg->header.stamp.toSec();

  // frequency control保证发给后端的图像频率不超过FREQ，即10Hz
  if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)
  {
    PUB_THIS_FRAME = true;
    // reset the frequency control当时间间隔比较大时，对于接收到的图像数目敏感性会下降，因此需要重置pub_count和first_image_time
    if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
    {
      first_image_time = img_msg->header.stamp.toSec();
      pub_count = 0;
    }
  }
  else
    PUB_THIS_FRAME = false;

  // 即使不发布也是要正常做光流追踪的，因为光流要求图像变化尽可能小
  // CvImageConstPtr是指向CvImage的常量指针，CvImage是一个模板类，它包含了一个图像和一个头部信息、编码信息等
  cv_bridge::CvImageConstPtr ptr;
  if (img_msg->encoding == "8UC1")
  {
    sensor_msgs::Image img;
    img.header = img_msg->header;
    img.height = img_msg->height;
    img.width = img_msg->width;
    img.is_bigendian = img_msg->is_bigendian;
    img.step = img_msg->step;
    img.data = img_msg->data;
    img.encoding = "8UC1"; // toCvCopy将ROS格式的图像消息转换为OpenCV格式的图像，拷贝图像数据
    ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
  }
  else
    ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

  cv::Mat show_img = ptr->image;
  TicToc t_r;
  for (int i = 0; i < NUM_OF_CAM; i++)
  {
    ROS_DEBUG("processing camera %d", i);
    if (i != 1 || !STEREO_TRACK) // 一个相机且不是双目跟踪
      trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec());
    else
    {
      if (EQUALIZE)
      {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
      }
      else
        trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
    }

#if SHOW_UNDISTORTION
  trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
  }

  for (unsigned int i = 0;; i++)
  {
    bool completed = false;
    for (int j = 0; j < NUM_OF_CAM; j++)
      if (j != 1 || !STEREO_TRACK)
        completed |= trackerData[j].updateID(i); // 更新特征点id
    if (!completed)
      break;
  }

  // 给后端发布数据
  if (PUB_THIS_FRAME)
  {
    pub_count++; // 发布的图像帧数加1
    // 组织要发布的数据
    sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
    // sensor_msgs::ChannelFloat32是一个消息类型，它包含一个名为values的float32数组和string类型的name（该通道的名称），这个数组的长度是可变的
    sensor_msgs::ChannelFloat32 id_of_point; // 这里构造的通道都没有name，只有values
    sensor_msgs::ChannelFloat32 u_of_point;
    sensor_msgs::ChannelFloat32 v_of_point;
    sensor_msgs::ChannelFloat32 velocity_x_of_point;
    sensor_msgs::ChannelFloat32 velocity_y_of_point;

    feature_points->header = img_msg->header;
    feature_points->header.frame_id = "world";

    vector<set<int>> hash_ids(NUM_OF_CAM);
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
      auto& un_pts = trackerData[i].cur_un_pts; // 去畸变的归一化坐标
      auto& cur_pts = trackerData[i].cur_pts; // 像素坐标
      auto& ids = trackerData[i].ids; // 特征点id
      auto& pts_velocity = trackerData[i].pts_velocity; // 特征点速度
      for (unsigned int j = 0; j < ids.size(); j++)
      {
        // 如果特征点被跟踪到了两次以上，那么就发布，因为等于1 没法构成重投影约束，也没法三角化
        if (trackerData[i].track_cnt[j] > 1)
        {
          int p_id = ids[j];
          hash_ids[i].insert(p_id); // 没用使用到
          geometry_msgs::Point32 p;
          p.x = un_pts[j].x;
          p.y = un_pts[j].y;
          p.z = 1;

          feature_points->points.push_back(p);
          id_of_point.values.push_back(float(p_id * NUM_OF_CAM + i));
          u_of_point.values.push_back(cur_pts[j].x);
          v_of_point.values.push_back(cur_pts[j].y);
          velocity_x_of_point.values.push_back(pts_velocity[j].x);
          velocity_y_of_point.values.push_back(pts_velocity[j].y);
        }
      }
    }
    // 点云数据结构，每个点可能有其x, y, z位置。此外，每个点也可能有与之关联的其他数据，如强度或颜色。这些额外的数据可以通过ChannelFloat32来表示
    feature_points->channels.push_back(id_of_point);
    feature_points->channels.push_back(u_of_point);
    feature_points->channels.push_back(v_of_point);
    feature_points->channels.push_back(velocity_x_of_point);
    feature_points->channels.push_back(velocity_y_of_point);
    ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
    // skip the first image; since no optical speed on first image
    if (!init_pub)
    {
      init_pub = true;
    }
    else
      pub_img.publish(feature_points); // 前端的得到的特征点数据发布给后端

    // 可视化相关的一些操作
    if (SHOW_TRACK)
    {
      ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
      //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
      cv::Mat stereo_img = ptr->image; // cv::Mat对象本身并不直接存储图像数据。它更像是一个智能指针或者头部信息，指向实际的图像数据
                                       // 这样赋值时，实际上是在复制 cv::Mat 对象（头部信息，引用计数等），而不是它指向的数据
      for (int i = 0; i < NUM_OF_CAM; i++)
      {
        cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
        cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

        for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
        {
          double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
          cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
        }
      }
      pub_match.publish(ptr->toImageMsg());
    }
  }
  ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "feature_tracker"); // ros节点初始化
  ros::NodeHandle n("~"); // 私有命名空间，会在发出的话题名字前加上节点名称
  // ros::console是ROS的日志管理模块，它提供了一套机制来输出和管理来自ROS节点的日志消息。日志输出等级包括调试、信息、警告、错误或致命
  // name：记录器的名称，可以是包名或者节点名，ROSCONSOLE_DEFAULT_NAME，当前节点的名称，当你想设置当前节点的日志级别时，你会使用这个宏
  // 日志的“信息”级别。ROS 的日志级别从最低到最高有：Debug, Info, Warn, Error, 和 Fatal。设置为 Info
  // 意味着所有低于 Info 的日志消息（如 Debug）将不会被显示，而 Info 及以上级别的消息将被显示
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
  readParameters(n); // 读取配置文件

  for (int i = 0; i < NUM_OF_CAM; i++)
    trackerData[i].readIntrinsicParameter(CAM_NAMES[i]); // 根据传入的配置文件构造相机模型

  if(FISHEYE)
  {
    for (auto &i: trackerData)
    {
      i.fisheye_mask = cv::imread(FISHEYE_MASK, 0);
      if(!i.fisheye_mask.data)
      {
        ROS_INFO("load mask fail");
        ROS_BREAK();
      }
      else
        ROS_INFO("load mask success");
    }
  }

  // 订阅图像话题
  ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);

  // 发布特征点相关的话题
  pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
  pub_match = n.advertise<sensor_msgs::Image>("feature_img", 1000);
  pub_restart = n.advertise<std_msgs::Bool>("restart", 1000);
  /*
  if (SHOW_TRACK)
      cv::namedWindow("vis", cv::WINDOW_NORMAL);
  */
  ros::spin();
  return 0;
}


// new points velocity is 0, pub or not?
// track cnt > 1 pub?