#include "camodocal/camera_models/CameraFactory.h"

#include <boost/algorithm/string.hpp>


#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/EquidistantCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "camodocal/camera_models/ScaramuzzaCamera.h"

#include "ceres/ceres.h"

namespace camodocal {

// 在C++中，静态成员变量需要在类声明之外进行定义和初始化
boost::shared_ptr<CameraFactory> CameraFactory::m_instance;
CameraFactory::CameraFactory() = default;

boost::shared_ptr<CameraFactory> CameraFactory::instance() {
  if (m_instance.get() == nullptr) {
    m_instance.reset(new CameraFactory); // 释放当前智能指针所管理的对象,并可以选择性地将智能指针指向一个新的对象
  }     // 参数p是新对象的原始指针

  return m_instance;
}

CameraPtr CameraFactory::generateCamera(Camera::ModelType modelType, const std::string& cameraName, 
                                        const cv::Size& imageSize) {
  switch (modelType) {
    case Camera::KANNALA_BRANDT: {
      EquidistantCameraPtr camera(new EquidistantCamera);

      EquidistantCamera::Parameters params = camera->getParameters();
      params.cameraName() = cameraName;
      params.imageWidth() = imageSize.width;
      params.imageHeight() = imageSize.height;
      camera->setParameters(params);
      return camera;
    }
    case Camera::PINHOLE: {
      PinholeCameraPtr camera(new PinholeCamera);

      PinholeCamera::Parameters params = camera->getParameters();
      params.cameraName() = cameraName;
      params.imageWidth() = imageSize.width;
      params.imageHeight() = imageSize.height;
      camera->setParameters(params);
      return camera;
    }
    case Camera::SCARAMUZZA: {
      OCAMCameraPtr camera(new OCAMCamera);

      OCAMCamera::Parameters params = camera->getParameters();
      params.cameraName() = cameraName;
      params.imageWidth() = imageSize.width;
      params.imageHeight() = imageSize.height;
      camera->setParameters(params);
      return camera;
    }
   case Camera::MEI:
   default: {
     CataCameraPtr camera(new CataCamera);

     CataCamera::Parameters params = camera->getParameters();
     params.cameraName() = cameraName;
     params.imageWidth() = imageSize.width;
     params.imageHeight() = imageSize.height;
     camera->setParameters(params);
     return camera;
  }
  }
}

CameraPtr CameraFactory::generateCameraFromYamlFile(const std::string& filename) {
  cv::FileStorage fs(filename, cv::FileStorage::READ);

  if (!fs.isOpened()) {
    return {};
  }

  Camera::ModelType modelType = Camera::MEI;
  if (!fs["model_type"].isNone()) {
    std::string sModelType;
    fs["model_type"] >> sModelType;
    // boost::iequals适用于不区分大小写的字符串比较
    if (boost::iequals(sModelType, "kannala_brandt")) {
      modelType = Camera::KANNALA_BRANDT;
    }
    else if (boost::iequals(sModelType, "mei")) {
      modelType = Camera::MEI;
    }
    else if (boost::iequals(sModelType, "scaramuzza")) {
      modelType = Camera::SCARAMUZZA;
    }
    else if (boost::iequals(sModelType, "pinhole")) {
      modelType = Camera::PINHOLE;
    }
    else {
      std::cerr << "# ERROR: Unknown camera model: " << sModelType << std::endl;
      return {};
    }
  }

  switch (modelType) {
    case Camera::KANNALA_BRANDT: {
      EquidistantCameraPtr camera(new EquidistantCamera);

      EquidistantCamera::Parameters params = camera->getParameters();
      params.readFromYamlFile(filename);
      camera->setParameters(params);
      return camera;
    }
    case Camera::PINHOLE: {
      PinholeCameraPtr camera(new PinholeCamera);

      PinholeCamera::Parameters params = camera->getParameters();
      params.readFromYamlFile(filename);
      camera->setParameters(params);
      return camera;
    }
    case Camera::SCARAMUZZA: {
      OCAMCameraPtr camera(new OCAMCamera);

      OCAMCamera::Parameters params = camera->getParameters();
      params.readFromYamlFile(filename);
      camera->setParameters(params);
      return camera;
    }
    case Camera::MEI:
    default: {
      CataCameraPtr camera(new CataCamera);

      CataCamera::Parameters params = camera->getParameters();
      params.readFromYamlFile(filename);
      camera->setParameters(params);
      return camera;
    }
    }

}

}

