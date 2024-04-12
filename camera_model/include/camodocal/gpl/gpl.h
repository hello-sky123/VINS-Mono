#ifndef GPL_H
#define GPL_H

#include <algorithm>
#include <cmath>
#include <random>
#include <opencv2/core/core.hpp>

namespace camodocal
{

template<class T>
T clamp(const T& v, const T& a, const T& b)
{
  return std::min(b, std::max(a, v));
}

double hypot3(double x, double y, double z);
float hypot3f(float x, float y, float z);

template<class T>
T normalizeTheta(const T& theta)
{
  T normTheta = theta;

  while (normTheta < - M_PI)
  {
      normTheta += 2.0 * M_PI;
  }
  while (normTheta > M_PI)
  {
      normTheta -= 2.0 * M_PI;
  }

  return normTheta;
}

double d2r(double deg);
float d2r(float deg);
double r2d(double rad);
float r2d(float rad);

double sinc(double theta);

template<class T>
T square(const T& x)
{
  return x * x;
}

template<class T>
T cube(const T& x)
{
  return x * x * x;
}

// 产生[a, b]之间均匀分布的随机数
template<class T>
T random(const T& a, const T& b)
{
  std::random_device RD;
  std::mt19937 RNG(RD());
  std::uniform_real_distribution<T> dist(a, b);
  return dist(RNG);
}

template<class T>
T randomNormal(const T& sigma) // 产生均值为0，方差为sigma的正态分布随机数(由Box-Muller方法产生)
{
  T x1, x2, w;
// 首先会执行一次do之内的语句，然后在while内检查条件是否为真，如果条件为真的话，就会重复do...while这个循环,直至while为假
  do
  {
    x1 = 2.0 * random(0.0, 1.0) - 1.0;
    x2 = 2.0 * random(0.0, 1.0) - 1.0;
    w = x1 * x1 + x2 * x2;
  }
  while (w >= 1.0 || w == 0.0);

  w = sqrt((-2.0 * log(w)) / w);

  return x1 * w * sigma;
}

unsigned long long timeInMicroseconds();

double timeInSeconds();

void colorDepthImage(cv::Mat& imgDepth, cv::Mat& imgColoredDepth, float minRange, float maxRange);

bool colormap(const std::string& name, unsigned char idx, float& r, float& g, float& b);

std::vector<cv::Point2i> bresLine(int x0, int y0, int x1, int y1);
std::vector<cv::Point2i> bresCircle(int x0, int y0, int r);

void fitCircle(const std::vector<cv::Point2d>& points, double& centerX, double& centerY, double& radius);

std::vector<cv::Point2d> intersectCircles(double x1, double y1, double r1, double x2, double y2, double r2);

void LLtoUTM(double latitude, double longitude, double& utmNorthing, double& utmEasting, std::string& utmZone);

void UTMtoLL(double utmNorthing, double utmEasting, const std::string& utmZone, double& latitude, double& longitude);

long timestampDiff(uint64_t t1, uint64_t t2);

}

#endif
