#include "camodocal/calib/CameraCalibration.h"

#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <utility>
#include <opencv2/imgproc/imgproc_c.h>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/sparse_graph/Transform.h"
#include "camodocal/gpl/EigenQuaternionParameterization.h"
#include "camodocal/camera_models/CostFunctionFactory.h"

#include "ceres/ceres.h"
namespace camodocal
{

CameraCalibration::CameraCalibration()
: m_boardSize(cv::Size(0,0)), m_squareSize(0.0f), m_verbose(false)
{ }

CameraCalibration::CameraCalibration(const Camera::ModelType modelType, const std::string& cameraName, const cv::Size& imageSize,
                                     cv::Size  boardSize, float squareSize)
: m_boardSize(std::move(boardSize)), m_squareSize(squareSize), m_verbose(false)
{
  m_camera = CameraFactory::instance()->generateCamera(modelType, cameraName, imageSize);
}

void CameraCalibration::clear() // 清空数据
{
  m_imagePoints.clear();
  m_scenePoints.clear();
}

void CameraCalibration::addChessboardData(const std::vector<cv::Point2f>& corners) // 添加单张图片的角点
{
  m_imagePoints.push_back(corners);

  std::vector<cv::Point3f> scenePointsInView;
  for (int i = 0; i < m_boardSize.height; ++i)
  {
    for (int j = 0; j < m_boardSize.width; ++j)
    {
      scenePointsInView.emplace_back(float(i) * m_squareSize, float(j) * m_squareSize, 0.0);
    }
  }
  m_scenePoints.push_back(scenePointsInView);
}

bool CameraCalibration::calibrate()
{
  int imageCount = int(m_imagePoints.size()); // 图片数量

  // compute intrinsic camera parameters and extrinsic parameters for each of the views
  std::vector<cv::Mat> rvecs;
  std::vector<cv::Mat> tvecs;
  bool ret = calibrateHelper(m_camera, rvecs, tvecs);

  m_cameraPoses = cv::Mat(imageCount, 6, CV_64F);
  for (int i = 0; i < imageCount; ++i)
  {
    m_cameraPoses.at<double>(i,0) = rvecs.at(i).at<double>(0);
    m_cameraPoses.at<double>(i,1) = rvecs.at(i).at<double>(1);
    m_cameraPoses.at<double>(i,2) = rvecs.at(i).at<double>(2);
    m_cameraPoses.at<double>(i,3) = tvecs.at(i).at<double>(0);
    m_cameraPoses.at<double>(i,4) = tvecs.at(i).at<double>(1);
    m_cameraPoses.at<double>(i,5) = tvecs.at(i).at<double>(2);
  }

  // Compute measurement covariance.
  std::vector<std::vector<cv::Point2f> > errVec(m_imagePoints.size());
  Eigen::Vector2d errSum = Eigen::Vector2d::Zero();
  size_t errCount = 0;
  for (size_t i = 0; i < m_imagePoints.size(); ++i)
  {
    std::vector<cv::Point2f> estImagePoints;
    m_camera->projectPoints(m_scenePoints.at(i), rvecs.at(i), tvecs.at(i),
                            estImagePoints);

    for (size_t j = 0; j < m_imagePoints.at(i).size(); ++j)
    {
      cv::Point2f pObs = m_imagePoints.at(i).at(j);
      cv::Point2f pEst = estImagePoints.at(j);

      cv::Point2f err = pObs - pEst;

      errVec.at(i).push_back(err);

      errSum += Eigen::Vector2d(err.x, err.y);
    }

    errCount += m_imagePoints.at(i).size();
  }

  Eigen::Vector2d errMean = errSum / static_cast<double>(errCount); // 计算误差均值（所有图像）

  Eigen::Matrix2d measurementCovariance = Eigen::Matrix2d::Zero();
  for (size_t i = 0; i < errVec.size(); ++i)
  {
    for (size_t j = 0; j < errVec.at(i).size(); ++j)
    {
      cv::Point2f err = errVec.at(i).at(j);
      double d0 = err.x - errMean(0);
      double d1 = err.y - errMean(1);

      measurementCovariance(0,0) += d0 * d0;
      measurementCovariance(0,1) += d0 * d1;
      measurementCovariance(1,1) += d1 * d1;
    }
  }
  measurementCovariance /= static_cast<double>(errCount);
  measurementCovariance(1,0) = measurementCovariance(0,1);

  m_measurementCovariance = measurementCovariance;

  return ret;
}

int CameraCalibration::sampleCount() const
{
  return (int)m_imagePoints.size();
}

std::vector<std::vector<cv::Point2f>>& CameraCalibration::imagePoints()
{
  return m_imagePoints;
}

const std::vector<std::vector<cv::Point2f>>& CameraCalibration::imagePoints() const
{
  return m_imagePoints;
}

std::vector<std::vector<cv::Point3f>>& CameraCalibration::scenePoints()
{
  return m_scenePoints;
}

const std::vector<std::vector<cv::Point3f>>& CameraCalibration::scenePoints() const
{
  return m_scenePoints;
}

CameraPtr& CameraCalibration::camera()
{
  return m_camera;
}

CameraConstPtr CameraCalibration::camera() const
{
  return m_camera;
}

Eigen::Matrix2d& CameraCalibration::measurementCovariance()
{
  return m_measurementCovariance;
}

const Eigen::Matrix2d& CameraCalibration::measurementCovariance() const
{
  return m_measurementCovariance;
}

cv::Mat& CameraCalibration::cameraPoses()
{
  return m_cameraPoses;
}

const cv::Mat& CameraCalibration::cameraPoses() const
{
  return m_cameraPoses;
}

void CameraCalibration::drawResults(std::vector<cv::Mat>& images) const
{
  std::vector<cv::Mat> rvecs, tvecs;

  for (size_t i = 0; i < images.size(); ++i)
  {
    cv::Mat rvec(3, 1, CV_64F);
    rvec.at<double>(0) = m_cameraPoses.at<double>(i,0);
    rvec.at<double>(1) = m_cameraPoses.at<double>(i,1);
    rvec.at<double>(2) = m_cameraPoses.at<double>(i,2);

    cv::Mat tvec(3, 1, CV_64F);
    tvec.at<double>(0) = m_cameraPoses.at<double>(i,3);
    tvec.at<double>(1) = m_cameraPoses.at<double>(i,4);
    tvec.at<double>(2) = m_cameraPoses.at<double>(i,5);

    rvecs.push_back(rvec);
    tvecs.push_back(tvec);
  }

  int drawShiftBits = 4;
  int drawMultiplier = 1 << drawShiftBits;

  cv::Scalar green(0, 255, 0);
  cv::Scalar red(0, 0, 255);

  for (size_t i = 0; i < images.size(); ++i)
  {
    cv::Mat& image = images.at(i);
    if (image.channels() == 1)
    {
      cv::cvtColor(image, image, CV_GRAY2RGB);
    }

    std::vector<cv::Point2f> estImagePoints;
    m_camera->projectPoints(m_scenePoints.at(i), rvecs.at(i), tvecs.at(i),
                            estImagePoints);

    float errorSum = 0.0f;
    float errorMax = std::numeric_limits<float>::min();

    for (size_t j = 0; j < m_imagePoints.at(i).size(); ++j)
    {
<<<<<<< HEAD
      cv::Point2f pObs = m_imagePoints.at(i).at(j);
      cv::Point2f pEst = estImagePoints.at(j);

      cv::circle(image,
                 cv::Point(cvRound(pObs.x * (float)drawMultiplier),
                           cvRound(pObs.y * (float)drawMultiplier)),
                 5, green, 2, CV_AA, drawShiftBits);

      cv::circle(image,
                 cv::Point(cvRound(pEst.x * (float)drawMultiplier),
                           cvRound(pEst.y * (float)drawMultiplier)),
                 5, red, 2, CV_AA, drawShiftBits);

      float error = (float)cv::norm(pObs - pEst);

      errorSum += error;
      if (error > errorMax)
      {
        errorMax = error;
      }
=======
        cv::Mat& image = images.at(i);
        if (image.channels() == 1)
        {
            cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
        }

        std::vector<cv::Point2f> estImagePoints;
        m_camera->projectPoints(m_scenePoints.at(i), rvecs.at(i), tvecs.at(i),
                                estImagePoints);

        float errorSum = 0.0f;
        float errorMax = std::numeric_limits<float>::min();

        for (size_t j = 0; j < m_imagePoints.at(i).size(); ++j)
        {
            cv::Point2f pObs = m_imagePoints.at(i).at(j);
            cv::Point2f pEst = estImagePoints.at(j);

            cv::circle(image,
                       cv::Point(cvRound(pObs.x * drawMultiplier),
                                 cvRound(pObs.y * drawMultiplier)),
                       5, green, 2, cv::LINE_AA, drawShiftBits);

            cv::circle(image,
                       cv::Point(cvRound(pEst.x * drawMultiplier),
                                 cvRound(pEst.y * drawMultiplier)),
                       5, red, 2, cv::LINE_AA, drawShiftBits);

            float error = cv::norm(pObs - pEst);

            errorSum += error;
            if (error > errorMax)
            {
                errorMax = error;
            }
        }

        std::ostringstream oss;
        oss << "Reprojection error: avg = " << errorSum / m_imagePoints.at(i).size()
            << "   max = " << errorMax;

        cv::putText(image, oss.str(), cv::Point(10, image.rows - 10),
                    cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255),
                    1, cv::LINE_AA);
>>>>>>> 90dabb5ec79946ae42fd2e1e91d4e69aabe1e25d
    }

    std::ostringstream oss;
    oss << "Reprojection error: avg = " << errorSum / (float)m_imagePoints.at(i).size()
        << "   max = " << errorMax;

    cv::putText(image, oss.str(), cv::Point(10, image.rows - 10),
                cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255),
                1, CV_AA);
  }
}

void CameraCalibration::writeParams(const std::string& filename) const
{
  m_camera->writeParametersToYamlFile(filename);
}

bool CameraCalibration::writeChessboardData(const std::string& filename) const
{
  std::ofstream ofs(filename.c_str(), std::ios::out | std::ios::binary);
  if (!ofs.is_open())
  {
    return false;
  }

  writeData(ofs, m_boardSize.width);
  writeData(ofs, m_boardSize.height);
  writeData(ofs, m_squareSize);

  writeData(ofs, m_measurementCovariance(0,0));
  writeData(ofs, m_measurementCovariance(0,1));
  writeData(ofs, m_measurementCovariance(1,0));
  writeData(ofs, m_measurementCovariance(1,1));

  writeData(ofs, m_cameraPoses.rows);
  writeData(ofs, m_cameraPoses.cols);
  writeData(ofs, m_cameraPoses.type());
  for (int i = 0; i < m_cameraPoses.rows; ++i)
  {
    for (int j = 0; j < m_cameraPoses.cols; ++j)
    {
      writeData(ofs, m_cameraPoses.at<double>(i, j));
    }
  }

  writeData(ofs, m_imagePoints.size());
  for (const auto & m_imagePoint: m_imagePoints)
  {
    writeData(ofs, m_imagePoint.size());
    for (const auto & ipt: m_imagePoint)
    {
      writeData(ofs, ipt.x);
      writeData(ofs, ipt.y);
    }
  }

  writeData(ofs, m_scenePoints.size());
  for (const auto & m_scenePoint: m_scenePoints)
  {
    writeData(ofs, m_scenePoint.size());
    for (const auto & spt: m_scenePoint)
    {
      writeData(ofs, spt.x);
      writeData(ofs, spt.y);
      writeData(ofs, spt.z);
    }
  }

  return true;
}

bool CameraCalibration::readChessboardData(const std::string& filename)
{
  std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
  if (!ifs.is_open())
  {
    return false;
  }

  readData(ifs, m_boardSize.width);
  readData(ifs, m_boardSize.height);
  readData(ifs, m_squareSize);

  readData(ifs, m_measurementCovariance(0, 0));
  readData(ifs, m_measurementCovariance(0, 1));
  readData(ifs, m_measurementCovariance(1, 0));
  readData(ifs, m_measurementCovariance(1, 1));

  int rows, cols, type;
  readData(ifs, rows);
  readData(ifs, cols);
  readData(ifs, type);
  m_cameraPoses = cv::Mat(rows, cols, type);

  for (int i = 0; i < m_cameraPoses.rows; ++i)
  {
    for (int j = 0; j < m_cameraPoses.cols; ++j)
    {
      readData(ifs, m_cameraPoses.at<double>(i, j));
    }
  }

  size_t nImagePointSets;
  readData(ifs, nImagePointSets);

  m_imagePoints.clear();
  m_imagePoints.resize(nImagePointSets);
  for (auto & m_imagePoint: m_imagePoints)
  {
    size_t nImagePoints;
    readData(ifs, nImagePoints);
    m_imagePoint.resize(nImagePoints);

    for (auto & ipt: m_imagePoint)
    {
      readData(ifs, ipt.x);
      readData(ifs, ipt.y);
    }
  }

  size_t nScenePointSets;
  readData(ifs, nScenePointSets);

  m_scenePoints.clear();
  m_scenePoints.resize(nScenePointSets);
  for (auto & m_scenePoint: m_scenePoints)
  {
    size_t nScenePoints;
    readData(ifs, nScenePoints);
    m_scenePoint.resize(nScenePoints);

    for (auto & spt: m_scenePoint)
    {
      readData(ifs, spt.x);
      readData(ifs, spt.y);
      readData(ifs, spt.z);
    }
  }

  return true;
}

void CameraCalibration::setVerbose(bool verbose)
{
  m_verbose = verbose;
}

bool CameraCalibration::calibrateHelper(CameraPtr& camera, std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs) const
{
  rvecs.assign(m_scenePoints.size(), cv::Mat()); // assign()方法用于给vector赋值
  tvecs.assign(m_scenePoints.size(), cv::Mat()); // 有3种重载形式，使用范围赋值，使用初始化列表赋值，使用n个相同值赋值

  // STEP 1: Estimate intrinsics（主要是求出f初值）
  camera->estimateIntrinsics(m_boardSize, m_scenePoints, m_imagePoints);

  // STEP 2: Estimate extrinsics（得到初步的外参）
  for (size_t i = 0; i < m_scenePoints.size(); ++i)
  {
    camera->estimateExtrinsics(m_scenePoints.at(i), m_imagePoints.at(i), rvecs.at(i), tvecs.at(i));
  }

  if (m_verbose)
  {
    std::cout << "[" << camera->cameraName() << "] "
              << "# INFO: " << "Initial reprojection error: "
              << std::fixed << std::setprecision(3)
              << camera->reprojectionError(m_scenePoints, m_imagePoints, rvecs, tvecs)
              << " pixels" << std::endl;
  }

  // STEP 3: optimization using ceres
  optimize(camera, rvecs, tvecs); // 创建一个优化问题

  if (m_verbose)
  {
    double err = camera->reprojectionError(m_scenePoints, m_imagePoints, rvecs, tvecs);
    std::cout << "[" << camera->cameraName() << "] " << "# INFO: Final reprojection error: "
              << err << " pixels" << std::endl;
    std::cout << "[" << camera->cameraName() << "] " << "# INFO: "
              << camera->parametersToString() << std::endl;
  }

  return true;
}

void CameraCalibration::optimize(CameraPtr& camera, std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs) const
{
  // Use ceres to do optimization
  ceres::Problem problem; // 定义优化问题

  std::vector<Transform, Eigen::aligned_allocator<Transform>> transformVec(rvecs.size());
  // 将cv格式的外参转换为Eigen格式，并将旋转矩阵和平移向量合并到一个Transform对象中
  for (size_t i = 0; i < rvecs.size(); ++i)
  {
    Eigen::Vector3d rvec;
    cv::cv2eigen(rvecs.at(i), rvec); // 将cv::Mat转换为Eigen::Vector3d

    transformVec.at(i).rotation() = Eigen::AngleAxisd(rvec.norm(), rvec.normalized());
    transformVec.at(i).translation() << tvecs[i].at<double>(0),
                                           tvecs[i].at<double>(1),
                                           tvecs[i].at<double>(2);
  }

  std::vector<double> intrinsicCameraParams;
  m_camera->writeParameters(intrinsicCameraParams); // 将相机内参写入intrinsicCameraParams，顺序为k1, k2, p1, p2, fx, fy, cx, cy

  // create residuals for each observation
  for (size_t i = 0; i < m_imagePoints.size(); ++i)
  {
    for (size_t j = 0; j < m_imagePoints.at(i).size(); ++j) // 遍历每张图片的每个角点
    {
      const cv::Point3f& spt = m_scenePoints.at(i).at(j);
      const cv::Point2f& ipt = m_imagePoints.at(i).at(j);

      ceres::CostFunction* costFunction =
          CostFunctionFactory::instance()->generateCostFunction(camera,Eigen::Vector3d(spt.x, spt.y, spt.z),
                                                                Eigen::Vector2d(ipt.x, ipt.y),
                                                                CAMERA_INTRINSICS | CAMERA_POSE);

      ceres::LossFunction* lossFunction = new ceres::CauchyLoss(1.0); // 添加核函数CauchyLoss，表达形式为pho(s) = a^2 * ln(1 + s/a^2)
      // 添加残差块
      problem.AddResidualBlock(costFunction, lossFunction,
                               intrinsicCameraParams.data(),
                               transformVec.at(i).rotationData(),
                               transformVec.at(i).translationData()); // 残差块的参数为代价函数、优化使用的鲁棒核函数、待优化变量
    }
    // 某些参数可能存在特定的结构或者约束，例如表示旋转的四元数，它们的模为1，当我们在优化过程中更新四元数时，我们需要确保这个约束始终得到满足
    // 为了处理这种情况，Ceres Solver提供了LocalParameterization类，它允许我们为参数定义局部更新规则。这确保了在优化过程中参数的特定结构或约束得到保持
    ceres::LocalParameterization* quaternionParameterization =
        new EigenQuaternionParameterization;

    problem.SetParameterization(transformVec.at(i).rotationData(),
                                quaternionParameterization); // 告诉Ceres在优化这个四元数时使用我们提供的局部参数化方法
  }

  std::cout << "begin ceres" << std::endl;
  ceres::Solver::Options options; // 配置求解器选项
  options.max_num_iterations = 1000; // 最大迭代次数

  if (m_verbose) // 是否输出优化过程
  {
    options.minimizer_progress_to_stdout = true;
  }

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem,  &summary);
  std::cout << "end ceres" << std::endl;

  if (m_verbose)
  {
    std::cout << summary.FullReport() << std::endl;
  }

  camera->readParameters(intrinsicCameraParams);

  for (size_t i = 0; i < rvecs.size(); ++i)
  {
    Eigen::AngleAxisd aa(transformVec.at(i).rotation());

    Eigen::Vector3d rvec = aa.angle() * aa.axis();
    cv::eigen2cv(rvec, rvecs.at(i));

    cv::Mat& tvec = tvecs.at(i);
    tvec.at<double>(0) = transformVec.at(i).translation()(0);
    tvec.at<double>(1) = transformVec.at(i).translation()(1);
    tvec.at<double>(2) = transformVec.at(i).translation()(2);
  }
}

template<typename T> // 读取数据
void CameraCalibration::readData(std::ifstream& ifs, T& data) const
{
  char* buffer = new char[sizeof(T)];

  ifs.read(buffer, sizeof(T));

  data = *(reinterpret_cast<T*>(buffer));

  delete[] buffer;
}

template<typename T>
void CameraCalibration::writeData(std::ofstream& ofs, T data) const
{
  char* pData = reinterpret_cast<char*>(&data);

  ofs.write(pData, sizeof(T));
}

}
