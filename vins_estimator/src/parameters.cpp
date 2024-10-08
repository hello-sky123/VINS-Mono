#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string IMU_TOPIC;
double ROW, COL;
double TD, TR;

template <typename T>
T readParam(ros::NodeHandle& n, std::string name)
{
  T ans;
  if (n.getParam(name, ans))
  {
    ROS_INFO_STREAM("Loaded " << name << ": " << ans);
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
  config_file = readParam<std::string>(n, "config_file");
  cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
  if(!fsSettings.isOpened())
  {
    std::cerr << "ERROR: Wrong path to settings" << std::endl;
  }

  fsSettings["imu_topic"] >> IMU_TOPIC;

  SOLVER_TIME = fsSettings["max_solver_time"]; // 单次优化最大求解时间 0.04s
  NUM_ITERATIONS = fsSettings["max_num_iterations"]; // 单次优化最大迭代次数 8
  MIN_PARALLAX = fsSettings["keyframe_parallax"]; // 关键帧间最小视差 10
  MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH; // 采用统一的虚拟焦距

  std::string OUTPUT_PATH;
  fsSettings["output_path"] >> OUTPUT_PATH;
  VINS_RESULT_PATH = OUTPUT_PATH + "/vins_result_no_loop.csv";
  std::cout << "result path: " << VINS_RESULT_PATH << std::endl;

  // create folder if not exists
  FileSystemHelper::createDirectoryIfNotExists(OUTPUT_PATH.c_str());

  std::ofstream fout(VINS_RESULT_PATH, std::ios::out); // 写入模式
  fout.close();

  // imu和图像相关参数
  ACC_N = fsSettings["acc_n"]; // 加速度计测量噪声的标准差
  ACC_W = fsSettings["acc_w"]; // 加速度计随机游走噪声的标准差
  GYR_N = fsSettings["gyr_n"];
  GYR_W = fsSettings["gyr_w"];
  G.z() = fsSettings["g_norm"]; // 重力的大小
  ROW = fsSettings["image_height"];
  COL = fsSettings["image_width"];
  ROS_INFO("ROW: %f COL: %f ", ROW, COL);

  ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"]; // camera frame to imu frame的外参的标志位：0 准确；1 初始猜测；2 未知，待标定
  if (ESTIMATE_EXTRINSIC == 2)
  {
    ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
    RIC.emplace_back(Eigen::Matrix3d::Identity());
    TIC.emplace_back(Eigen::Vector3d::Zero());
    EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
  }
  else
  {
    if (ESTIMATE_EXTRINSIC == 1)
    {
      ROS_WARN("Optimize extrinsic param around initial guess!");
      EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
    }
    if (ESTIMATE_EXTRINSIC == 0)
      ROS_WARN("Fix extrinsic param!");

    cv::Mat cv_R, cv_T;
    fsSettings["extrinsicRotation"] >> cv_R;
    fsSettings["extrinsicTranslation"] >> cv_T;
    Eigen::Matrix3d eigen_R;
    Eigen::Vector3d eigen_T;
    cv::cv2eigen(cv_R, eigen_R); // 将cv::Mat转换为Eigen::Matrix
    cv::cv2eigen(cv_T, eigen_T);
    Eigen::Quaterniond Q(eigen_R); // 旋转矩阵转四元数
    eigen_R = Q.normalized(); // 四元数归一化
    RIC.push_back(eigen_R); // RIC和TIC是相机坐标系到IMU坐标系的外参
    TIC.push_back(eigen_T);
    ROS_INFO_STREAM("Extrinsic_R : " << std::endl << RIC[0]);
    ROS_INFO_STREAM("Extrinsic_T : " << std::endl << TIC[0].transpose());
  }

  INIT_DEPTH = 5.0; // 特征点深度的默认值
  BIAS_ACC_THRESHOLD = 0.1; // 没用到
  BIAS_GYR_THRESHOLD = 0.1;

  TD = fsSettings["td"]; // 相机和imu的时延
  ESTIMATE_TD = fsSettings["estimate_td"];
  if (ESTIMATE_TD)
    ROS_INFO_STREAM("asynchronous sensors, online estimate time offset, initial td: " << TD);
  else
    ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

  ROLLING_SHUTTER = fsSettings["rolling_shutter"]; // 是否是全局快门
  if (ROLLING_SHUTTER)
  {
    TR = fsSettings["rolling_shutter_tr"];
    ROS_INFO_STREAM("rolling shutter camera, read out time per line: " << TR);
  }
  else
  {
    TR = 0;
  }

  fsSettings.release();
}
