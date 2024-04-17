#include "camodocal/camera_models/Camera.h"
#include "camodocal/camera_models/ScaramuzzaCamera.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <utility> // 头文件中定义了一些实用的模板类和函数，如 pair、tuple、swap、forward 等

namespace camodocal {

Camera::Parameters::Parameters(ModelType modelType)
:  m_modelType(modelType),
   m_imageWidth(0),
   m_imageHeight(0)
{
  switch (modelType) {
  case KANNALA_BRANDT: // 在C++中，fallthrough是指在switch语句中，当一个case执行完毕后，接着执行下一个case，而不管下一个case是什么。
  case PINHOLE:
    m_nIntrinsics = 8;
    break;
  case SCARAMUZZA:
    m_nIntrinsics = SCARAMUZZA_CAMERA_NUM_PARAMS;
    break;
  case MEI:
  default:
    m_nIntrinsics = 9;
  }
}

Camera::Parameters::Parameters(ModelType modelType, std::string  cameraName, int w, int h)
:  m_modelType(modelType),
   m_cameraName(std::move(cameraName)),
   m_imageWidth(w),
   m_imageHeight(h)
{
  switch (modelType) {
  case KANNALA_BRANDT:
  case PINHOLE:
    m_nIntrinsics = 8;
    break;
  case SCARAMUZZA:
    m_nIntrinsics = SCARAMUZZA_CAMERA_NUM_PARAMS;
    break;
  case MEI:
  default:
    m_nIntrinsics = 9;
  }
}

Camera::ModelType& Camera::Parameters::modelType() {
  return m_modelType;
}

std::string& Camera::Parameters::cameraName() {
  return m_cameraName;
}

int& Camera::Parameters::imageWidth() {
  return m_imageWidth;
}

int& Camera::Parameters::imageHeight() {
  return m_imageHeight;
}

Camera::ModelType Camera::Parameters::modelType() const {
  return m_modelType;
}

const std::string& Camera::Parameters::cameraName() const {
  return m_cameraName;
}

int Camera::Parameters::imageWidth() const {
  return m_imageWidth;
}

int Camera::Parameters::imageHeight() const {
  return m_imageHeight;
}

int Camera::Parameters::nIntrinsics() const {
  return m_nIntrinsics;
}

cv::Mat& Camera::mask() {
  return m_mask;
}

const cv::Mat& Camera::mask() const {
  return m_mask;
}

// 估计相机外参与相机模型无关，因此在基类中实现
void Camera::estimateExtrinsics(const std::vector<cv::Point3f>& objectPoints,
                                const std::vector<cv::Point2f>& imagePoints,
                                cv::Mat& rvec, cv::Mat& tvec) const {
  std::vector<cv::Point2f> Ms(imagePoints.size()); //像素坐标
  for (size_t i = 0; i < Ms.size(); ++i) {
    Eigen::Vector3d P;  // std::vector提供了两种方法来访问元素：operator[] 和 at()。operator[] 不检查索引是否有效，而 at() 会检查索引是否有效。
    liftProjective(Eigen::Vector2d(imagePoints.at(i).x, imagePoints.at(i).y), P); // 反向投影得到归一化平面坐标，并去除畸变
    // liftProjective反向投影，将像素坐标转换为空间中对应点的坐标（lift提升）
    P /= P(2);

    Ms.at(i).x = P(0);
    Ms.at(i).y = P(1);
  }

  // assume unit focal length, zero principal point, and zero distortion
  // empty()是Mat类的成员函数，用来判断当前对象是否为空（没有行或者列），如果为空返回true，否则返回false，solvePnP默认使用的是迭代法求解
  cv::solvePnP(objectPoints, Ms, cv::Mat::eye(3, 3, CV_64F), cv::noArray(), rvec, tvec); // Tcw
}

double Camera::reprojectionDist(const Eigen::Vector3d& P1, const Eigen::Vector3d& P2) const {
  Eigen::Vector2d p1, p2;

  spaceToPlane(P1, p1);
  spaceToPlane(P2, p2);

  return (p1 - p2).norm();
}

double Camera::reprojectionError(const std::vector< std::vector<cv::Point3f> >& objectPoints,
                                 const std::vector< std::vector<cv::Point2f> >& imagePoints,
                                 const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
                                 cv::OutputArray _perViewErrors) const {
  int imageCount = objectPoints.size();
  size_t pointsSoFar = 0;
  double totalErr = 0.0;

  bool computePerViewErrors = _perViewErrors.needed();
  cv::Mat perViewErrors;
  if (computePerViewErrors) {
    _perViewErrors.create(imageCount, 1, CV_64F);
    perViewErrors = _perViewErrors.getMat();
  }

  for (int i = 0; i < imageCount; ++i) {
    size_t pointCount = imagePoints.at(i).size();

    pointsSoFar += pointCount;

    std::vector<cv::Point2f> estImagePoints;
    projectPoints(objectPoints.at(i), rvecs.at(i), tvecs.at(i),
                  estImagePoints);

    double err = 0.0;
    for (size_t j = 0; j < imagePoints.at(i).size(); ++j) {
      err += cv::norm(imagePoints.at(i).at(j) - estImagePoints.at(j));
    }

    if (computePerViewErrors) {
      perViewErrors.at<double>(i) = err / pointCount;
    }

    totalErr += err;
  }

  return totalErr / pointsSoFar;
}

double
Camera::reprojectionError(const Eigen::Vector3d& P,
                          const Eigen::Quaterniond& camera_q,
                          const Eigen::Vector3d& camera_t,
                          const Eigen::Vector2d& observed_p) const
{
    Eigen::Vector3d P_cam = camera_q.toRotationMatrix() * P + camera_t;

    Eigen::Vector2d p;
    spaceToPlane(P_cam, p);

    return (p - observed_p).norm();
}

void Camera::projectPoints(const std::vector<cv::Point3f>& objectPoints,
                           const cv::Mat& rvec, const cv::Mat& tvec,
                           std::vector<cv::Point2f>& imagePoints) const
{
  // project 3D object points to the image plane
  imagePoints.reserve(objectPoints.size());

  // double
  cv::Mat R0;
  cv::Rodrigues(rvec, R0);

  Eigen::MatrixXd R(3,3);
  R << R0.at<double>(0,0), R0.at<double>(0,1), R0.at<double>(0,2),
       R0.at<double>(1,0), R0.at<double>(1,1), R0.at<double>(1,2),
       R0.at<double>(2,0), R0.at<double>(2,1), R0.at<double>(2,2);

  Eigen::Vector3d t;
  t << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);

  for (size_t i = 0; i < objectPoints.size(); ++i)
  {
    const cv::Point3f& objectPoint = objectPoints.at(i);

    // Rotate and translate
    Eigen::Vector3d P;
    P << objectPoint.x, objectPoint.y, objectPoint.z;

    P = R * P + t;

    Eigen::Vector2d p;
    spaceToPlane(P, p);

    imagePoints.push_back(cv::Point2f(p(0), p(1)));
  }
}

}
