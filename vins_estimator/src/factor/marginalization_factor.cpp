#include "marginalization_factor.h"

/**
 * @brief 待边缘化的各个参数块计算残差和雅可比，同时处理核函数
 */
void ResidualBlockInfo::Evaluate()
{
  residuals.resize(cost_function->num_residuals()); // 残差的维度

  // 确定相关参数块的数目，比如预积分涉及到的参数块有4个，所以vector的size是4，存的元素是参数快的维度
  std::vector<int> block_sizes = cost_function->parameter_block_sizes();
  raw_jacobians = new double* [block_sizes.size()]; // ceres接口都是double数组，因此这里给雅可比准备数组
  jacobians.resize(block_sizes.size());

  // 这里把jacobians的每个矩阵地址赋给raw_jacobians，然后把raw_jacobians传递给ceres的接口，这样计算结果就会直接写入jacobians中
  for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
  {
    jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
    raw_jacobians[i] = jacobians[i].data(); // 取出指针
  }

  // 调用各自重载的Evaluate函数，计算残差和雅可比
  cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

  if (loss_function)
  {
    double residual_scaling_, alpha_sq_norm_;

    double sq_norm, rho[3];

    sq_norm = residuals.squaredNorm(); // 残差的模的平方
    loss_function->Evaluate(sq_norm, rho);  // pho[0]是核函数值，pho[1]是核函数的一阶导数，pho[2]是核函数的二阶导数

    double sqrt_rho1_ = sqrt(rho[1]);

    if ((sq_norm == 0.0) || (rho[2] <= 0.0)) // 满足这个条件，说明核函数给的符合要求
    {
      residual_scaling_ = sqrt_rho1_;
      alpha_sq_norm_ = 0.0;
    }
    else
    {
      const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
      const double alpha = 1.0 - sqrt(D);
      residual_scaling_ = sqrt_rho1_ / (1 - alpha);
      alpha_sq_norm_ = alpha / sq_norm;
    }

    // 这里相当于残差和雅可比都乘以sqrt_rho1_，即核函数所在点的一阶导数
    for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
    {
      jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
    }

    residuals *= residual_scaling_; // outlier区域，核函数在s处的一阶导小于1，残差会被缩小
  }
}

MarginalizationInfo::~MarginalizationInfo()
{   
  for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
    delete[] it->second;

  for (int i = 0; i < (int)factors.size(); i++)
  {
    delete[] factors[i]->raw_jacobians;
    delete factors[i]->cost_function;
    delete factors[i];
  }
}

/**
 * @brief 收集各个残差
 * @param[in] residual_block_info
 */
void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo* residual_block_info)
{
  factors.emplace_back(residual_block_info); // 将ResidualBlockInfo指针放入factors中

  std::vector<double*>& parameter_blocks = residual_block_info->parameter_blocks; // 这个是和该约束相关的参数块
  std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes(); // 各个参数块的大小

  for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++)
  {
    double* addr = parameter_blocks[i];
    int size = parameter_block_sizes[i];
    // 这里是个unordered_map，避免重复添加
    parameter_block_size[reinterpret_cast<long>(addr)] = size; // 地址->global size
  }

  // 待边缘化的参数块
  for (int i: residual_block_info->drop_set)
  {
    double* addr = parameter_blocks[i];
    // 先准备好待边缘化的参数块的unordered_map
    parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
  }
}

/**
 * @brief 将各个残差块计算残差和雅克比，同时备份所有相关的参数块内容
 */
void MarginalizationInfo::preMarginalize()
{
  for (auto it: factors) // 遍历所有的残差块
  {
    it->Evaluate(); // 调用这个接口计算各个残差块的残差和雅克比

    std::vector<int> block_sizes = it->cost_function->parameter_block_sizes(); // 得到每个残差块的参数块大小
    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
    {
      long addr = reinterpret_cast<long>(it->parameter_blocks[i]); // 每一个参数块的地址
      int size = block_sizes[i]; // 每一个参数块的大小
      // 把每一个参数块都备份起来，使用unordered_map避免参数块重复，之所以备份，是为了后面的状态保留
      if (parameter_block_data.find(addr) == parameter_block_data.end())
      {
        auto* data = new double[size];
        memcpy(data, it->parameter_blocks[i], sizeof(double) * size);
        parameter_block_data[addr] = data; // 地址->参数块实际内容的地址
      }
    }
  }
}

int MarginalizationInfo::localSize(int size)
{
  return size == 7 ? 6 : size;
}

int MarginalizationInfo::globalSize(int size) const
{
    return size == 6 ? 7 : size;
}

void* ThreadsConstructA(void* threadsstruct)
{
  ThreadsStruct* p = ((ThreadsStruct*)threadsstruct);
  
  // 遍历分配过来的任务
  for (auto it: p->sub_factors)
  {
    for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
    {
      int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])]; // 在大矩阵A中的索引
      int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])]; // 参数块的大小
      if (size_i == 7)
        size_i = 6;
      Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i); // 四元数是过参数化的，求导是用的是李代数，因此要取左边6列，第7列是0
      // 和本身以及其他雅可比块构造H矩阵，i是当前参数块，j是其他参数块
      for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
      {
        int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
        int size_j = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])];
        if (size_j == 7)
          size_j = 6;
        Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
        if (i == j)
          p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
        else
        {
          p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
          p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
        }
      }
      p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals; // 正常情况下 g = -JT * e
    }
  }
  return threadsstruct;
}

/**
 * @brief 边缘化处理，并将结果转换成残差和雅可比的形式
 */
void MarginalizationInfo::marginalize()
{
  int pos = 0;
  // parameter_block_idx是unordered_map，key是参数块地址，value初始都是0
  for (auto& it: parameter_block_idx)
  {
    it.second = pos; // 这是所有参数块中排序的id，待边缘化的参数块排在前面
    pos += localSize(parameter_block_size[it.first]); // 因为要求导，所以是localSize，具体一点就是使用李代数
  }

  m = pos; // 待边缘化的参数块的总大小（不是个数）

  for (const auto& it: parameter_block_size) // 遍历所有的参数块
  {
    if (parameter_block_idx.find(it.first) == parameter_block_idx.end()) // 如果不是待边缘化的参数块
    {
      parameter_block_idx[it.first] = pos;
      pos += localSize(it.second);
    }
  }

  n = pos - m; // 保留下来的参数块的总大小

  TicToc t_summing;
  Eigen::MatrixXd A(pos, pos); // Ax = b 预设大小
  Eigen::VectorXd b(pos);
  A.setZero();
  b.setZero();

  // 往矩阵A和向量b中填充数据，利用多线程加速
  TicToc t_thread_summing;
  pthread_t tids[NUM_THREADS]; // pthread_t 是 POSIX 线程库中用于标识线程id的数据类型
  ThreadsStruct threadsstruct[NUM_THREADS]; // 用于传递给线程的结构体数组
  int i = 0;
  for (auto it: factors)
  {
    threadsstruct[i].sub_factors.push_back(it); // 将残差块分配给不同的线程
    i++;
    i = i % NUM_THREADS;
  }

  // 每一个线程构造一个A和b，最后再合并
  for (int i = 0; i < NUM_THREADS; i++)
  {
    TicToc zero_matrix;
    // 所以 矩阵A和向量b的大小都是pos，初始值都是0
    threadsstruct[i].A = Eigen::MatrixXd::Zero(pos,pos);
    threadsstruct[i].b = Eigen::VectorXd::Zero(pos);
    // 多线程访问会带来竞争，因此需要将数据复制到结构体中，然后传递给线程
    threadsstruct[i].parameter_block_size = parameter_block_size; // 大小
    threadsstruct[i].parameter_block_idx = parameter_block_idx; // 索引
    // 当你创建一个新的线程时（例如，使用 pthread_create 函数），一个 pthread_t 类型的变量会被用作线程 ID 的存放位置
    // 第3个参数是线程要执行函数的地址，第4个参数是传递给线程函数的参数
    int ret = pthread_create(&tids[i], nullptr, ThreadsConstructA ,(void*)&(threadsstruct[i]));
    if (ret != 0)
    {
      ROS_WARN("pthread_create error");
      ROS_BREAK();
    }
  }

  for( int i = NUM_THREADS - 1; i >= 0; i--)
  {
    // 等待各个线程结束
    pthread_join(tids[i], nullptr);
    A += threadsstruct[i].A;
    b += threadsstruct[i].b;
  }

  //TODO
  // Amm矩阵的构建是为了保证正定性，正常情况下，Amm本身已经是正定的，但是由于数值误差，可能会导致不是正定的，因此需要做一些处理
  Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose()); // lambda_a
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm); // 实对称矩阵或者复厄米特矩阵的特征值分解

  // 一个逆矩阵的特征值是其原矩阵的特征值的倒数，特征向量不变，利用特征值取逆来构造逆矩阵，select类似于C++里的？：
  Eigen::MatrixXd Amm_inv = saes.eigenvectors() 
                          * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() 
                          * saes.eigenvectors().transpose();

  Eigen::VectorXd bmm = b.segment(0, m); // 待边缘化的大小
  Eigen::MatrixXd Amr = A.block(0, m, m, n); // lambda_b
  Eigen::MatrixXd Arm = A.block(m, 0, n, m);
  Eigen::MatrixXd Arr = A.block(m, m, n, n); // lambda_c
  Eigen::VectorXd brr = b.segment(m, n);
  A = Arr - Arm * Amm_inv * Amr;
  b = brr - Arm * Amm_inv * bmm;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
  Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
  Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

  Eigen::VectorXd S_sqrt = S.cwiseSqrt(); // 这个求得的是S^0.5，这里是向量，逐元素开方
  Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

  // 边缘化为了实现对剩余参数的约束，为了便于一起优化，将边缘化后的约束转换为残差和雅可比的形式
  linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
  linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
}

std::vector<double*> MarginalizationInfo::getParameterBlocks(std::unordered_map<long, double*>& addr_shift)
{
  std::vector<double*> keep_block_addr;
  keep_block_size.clear();
  keep_block_idx.clear();
  keep_block_data.clear();

  for (const auto& it: parameter_block_idx) // 遍历边缘化相关的每个参数块
  {
    if (it.second >= m) // 保留的参数块
    {
      keep_block_size.push_back(parameter_block_size[it.first]); // 留下来的参数块的大小 global size
      keep_block_idx.push_back(parameter_block_idx[it.first]); // 留下来的参数块的索引
      keep_block_data.push_back(parameter_block_data[it.first]); // 留下来的参数块的内容
      keep_block_addr.push_back(addr_shift[it.first]); // 对应的新地址
    }
  }
  // 保留的参数块的总大小
  sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);

  return keep_block_addr;
}

/**
 * @brief 边缘化信息结果的构造函数，根据边缘化信息确定参数块总数和残差维数
 * @param _marginalization_info
 */
MarginalizationFactor::MarginalizationFactor(MarginalizationInfo* _marginalization_info): marginalization_info(_marginalization_info)
{
  int cnt = 0;
  for (auto it: marginalization_info->keep_block_size) // keep_block_size表示上一次边缘化保留的参数块的大小
  {
    mutable_parameter_block_sizes()->push_back(it); // 这里是ceres的接口，用于确定参数块的大小
    cnt += it;
  }
  
  set_num_residuals(marginalization_info->n); // 这里是ceres的接口，用于确定残差的维度，local size
};

/**
 * @brief 边缘化结果的残差和雅可比的计算
 * @param[in] parameters
 * @param[out] residuals
 * @param[out] jacobians
 * @return bool
 */
bool MarginalizationFactor::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
{
  int n = marginalization_info->n;
  int m = marginalization_info->m; // 边缘化掉的参数块的大小
  Eigen::VectorXd dx(n);
  for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
  {
    int size = marginalization_info->keep_block_size[i];
    int idx = marginalization_info->keep_block_idx[i] - m; // idx起点统一为0
    Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size); // 当前调整后的参数块的值
    Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size); // 上一次边缘化保留的参数块的值
    if (size != 7)
      dx.segment(idx, size) = x - x0; // 不需要local parameterization的，直接作差
    else
    {
      dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
      dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() 
                             * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
      if ((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() *
           Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() < 0)
      {
        dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() 
                               * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
      }
    }
  }

  // 更新残差，边缘化后的先验误差 e = e0 + J * dx，根据FEJ，雅可比保持不变，但是残差随着优化会变化，因此下面只更新残差，不更新雅可比
  // 详情见https://www.zhihu.com/question/52869487、https://blog.csdn.net/weixin_41394379/article/details/89975386
  Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;
  if (jacobians)
  {
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
    {
      if (jacobians[i])
      {
        int size = marginalization_info->keep_block_size[i], local_size = marginalization_info->localSize(size);
        int idx = marginalization_info->keep_block_idx[i] - m;
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], n, size);
        jacobian.setZero();
        jacobian.leftCols(local_size) = marginalization_info->linearized_jacobians.middleCols(idx, local_size);
      }
    }
  }
  return true;
}
