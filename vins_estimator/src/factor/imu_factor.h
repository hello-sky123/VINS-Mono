#pragma once
#include <ros/assert.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "../utility/utility.h"
#include "../parameters.h"
#include "integration_base.h"

#include <ceres/ceres.h>

// 预积分的残差是15维，约束的是两帧之间的位姿和速度，以及两帧之间的陀螺仪零偏和加速度计零偏
class IMUFactor: public ceres::SizedCostFunction<15, 7, 9, 7, 9>
{
 public:
   IMUFactor() = delete;
   explicit IMUFactor(IntegrationBase* _pre_integration): pre_integration(_pre_integration)
   { }
   
   // parameters: 优化变量，double const* const*用于指向一个二维数组，有两种指针的方式指向二维数组，一种是数组指针，另一种是指针数组
   // 1. 数组指针：double (*parameters)[7]，指向一个数组，7代表了二维数组的列数，数组中每个元素是一个double类型的指针，指向了每一行行首，
   // 可以直接使用二维数组的数组名来初始化数组指针，如double a[2][3] = {{1, 2, 3}, {4, 5, 6}}; double (*p)[3] = a;
   // 2. 指针数组：double* parameters[7]，数组中每个元素是一个double类型的指针，可以用每一行的行首来初始化指针数组，
   // 如double a[2][3] = {{1, 2, 3}, {4, 5, 6}}; double* p[2] = {a[0], a[1]}; double** pp = p; 这样pp就指向了二维数组的首地址
   // 访问二维数组的元素时，可以使用指针的指针，也可以使用数组的数组名，如p[0][0], a[0][0];
   /**
    * @brief 使用解析求导，必须重载这个函数
    * @param parameters
    * @param residuals
    * @param jacobians
    * @return
    */
   bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
   {

     Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
     Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

     Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
     Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
     Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

     Eigen::Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
     Eigen::Quaterniond Qj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

     Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
     Eigen::Vector3d Baj(parameters[3][3], parameters[3][4], parameters[3][5]);
     Eigen::Vector3d Bgj(parameters[3][6], parameters[3][7], parameters[3][8]);

     #if 0
       if ((Bai - pre_integration->linearized_ba).norm() > 0.10 ||
          (Bgi - pre_integration->linearized_bg).norm() > 0.01)
       {
         pre_integration->repropagate(Bai, Bgi);
       }
     #endif

     Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
     // 得到残差
     residual = pre_integration->evaluate(Pi, Qi, Vi, Bai, Bgi, Pj, Qj, Vj, Baj, Bgj);

     // Eigen::LLT 是 Eigen 库中用于执行 Cholesky 分解的类。Cholesky 分解是一种将一个对称正定矩阵分解为一个下三角矩阵和其转置之积的方法
     // 对于大型矩阵，Cholesky 分解通常比其他方法（如 LU 分解或 QR 分解）更快，但它的使用受到矩阵必须是对称正定的限制
     Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration->covariance.inverse()).matrixL().transpose();
     // 信息矩阵等于协方差矩阵的逆，
     //sqrt_info.setIdentity();
     residual = sqrt_info * residual; // L^T * r 将信息矩阵乘以残差，得到加权的残差

     if (jacobians)
     {
       double sum_dt = pre_integration->sum_dt;
       Eigen::Matrix3d dp_dba = pre_integration->jacobian.template block<3, 3>(O_P, O_BA);
       Eigen::Matrix3d dp_dbg = pre_integration->jacobian.template block<3, 3>(O_P, O_BG);

       Eigen::Matrix3d dq_dbg = pre_integration->jacobian.template block<3, 3>(O_R, O_BG);

       Eigen::Matrix3d dv_dba = pre_integration->jacobian.template block<3, 3>(O_V, O_BA);
       Eigen::Matrix3d dv_dbg = pre_integration->jacobian.template block<3, 3>(O_V, O_BG);

       if (pre_integration->jacobian.maxCoeff() > 1e8 || pre_integration->jacobian.minCoeff() < -1e8)
       {
         ROS_WARN("numerical unstable in preintegration");
       }

       if (jacobians[0])
       {
         Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
         jacobian_pose_i.setZero();

         jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();
         jacobian_pose_i.block<3, 3>(O_P, O_R) = Utility::skewSymmetric(Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));

       #if 0
         jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Qj.inverse() * Qi).toRotationMatrix();
       #else
         Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
         jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Utility::Qleft(Qj.inverse() * Qi) * Utility::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();
       #endif

         jacobian_pose_i.block<3, 3>(O_V, O_R) = Utility::skewSymmetric(Qi.inverse() * (G * sum_dt + Vj - Vi));

         jacobian_pose_i = sqrt_info * jacobian_pose_i;

         if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
         {
           ROS_WARN("numerical unstable in preintegration");
         }
       }

       if (jacobians[1])
       {
         Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[1]);
         jacobian_speedbias_i.setZero();
         jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;
         jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
         jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;

       #if 0
          jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -dq_dbg;
       #else
          jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(Qj.inverse() * Qi * pre_integration->delta_q).bottomRightCorner<3, 3>() * dq_dbg;
       #endif

          jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) = -Qi.inverse().toRotationMatrix();
          jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
          jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;

          jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) = -Eigen::Matrix3d::Identity();

          jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) = -Eigen::Matrix3d::Identity();

          jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;
       }

       if (jacobians[2])
       {
         Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[2]);
         jacobian_pose_j.setZero();

         jacobian_pose_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();

       #if 0
         jacobian_pose_j.block<3, 3>(O_R, O_R) = Eigen::Matrix3d::Identity();
       #else
         Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
         jacobian_pose_j.block<3, 3>(O_R, O_R) = Utility::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
       #endif

         jacobian_pose_j = sqrt_info * jacobian_pose_j;
       }

       if (jacobians[3])
       {
         Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(jacobians[3]);
         jacobian_speedbias_j.setZero();

         jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) = Qi.inverse().toRotationMatrix();

         jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix3d::Identity();

         jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity();

         jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;
       }
     }

     return true;
   }

   IntegrationBase* pre_integration;

};

