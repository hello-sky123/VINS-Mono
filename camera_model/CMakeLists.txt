cmake_minimum_required(VERSION 3.5.1)
project(camera_model)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)

find_package(Boost REQUIRED COMPONENTS filesystem program_options system)
include_directories(${Boost_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})


# 环境配置：catkin_package() 宏负责生成和配置必要的环境设置文件（如pkg-config文件和CMake文件），这些文件在构建时间和运行时间被用来查找和链接依赖项。
# 依赖传递：通过指定 CATKIN_DEPENDS，Catkin构建系统会自动处理和传递依赖关系。这意味着当你构建一个依赖于当前包的其他包时，Catkin会确保所有依赖的头文件和库都是可用的。
# 跨包链接：通过 LIBRARIES 参数，catkin_package() 确保当其他包链接到这些库时，链接器可以找到正确的库文件
catkin_package(
  INCLUDE_DIRS include            # 指定包含目录
  LIBRARIES camera_model          # 声明该包提供的库
  CATKIN_DEPENDS roscpp std_msgs  # 声明该包依赖的其他catkin包
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

include_directories("include")


add_executable(Calibration 
	src/intrinsic_calib.cc
	src/chessboard/Chessboard.cc
	src/calib/CameraCalibration.cc
	src/camera_models/Camera.cc
	src/camera_models/CameraFactory.cc
	src/camera_models/CostFunctionFactory.cc
	src/camera_models/PinholeCamera.cc
	src/camera_models/CataCamera.cc
	src/camera_models/EquidistantCamera.cc
	src/camera_models/ScaramuzzaCamera.cc
	src/sparse_graph/Transform.cc
	src/gpl/gpl.cc
	src/gpl/EigenQuaternionParameterization.cc)

add_library(camera_model
	src/chessboard/Chessboard.cc
	src/calib/CameraCalibration.cc
	src/camera_models/Camera.cc
	src/camera_models/CameraFactory.cc
	src/camera_models/CostFunctionFactory.cc
	src/camera_models/PinholeCamera.cc
	src/camera_models/CataCamera.cc
	src/camera_models/EquidistantCamera.cc
	src/camera_models/ScaramuzzaCamera.cc
	src/sparse_graph/Transform.cc
	src/gpl/gpl.cc
	src/gpl/EigenQuaternionParameterization.cc)

target_link_libraries(Calibration ${Boost_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_link_libraries(camera_model ${Boost_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
