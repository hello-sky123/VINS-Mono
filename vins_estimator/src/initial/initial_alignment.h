#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "../factor/imu_factor.h"
#include "../utility/utility.h"
#include <ros/ros.h>
#include <map>
#include "../feature_manager.h"

using namespace Eigen;
using namespace std;

class ImageFrame
{
 public:
   ImageFrame() = default;
   ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& _points, double _t):
    t{_t}, is_key_frame{false}
   {
     points = _points;
   };
   map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> points; // 特征点的id，观测信息
   double t; // 这一帧的时间戳
   Matrix3d R; // 这一帧的位姿
   Vector3d T;
   IntegrationBase* pre_integration;
   bool is_key_frame; // 是否是关键帧
};

bool VisualIMUAlignment(map<double, ImageFrame>& all_image_frame, Vector3d* Bgs, Vector3d& g, VectorXd& x);