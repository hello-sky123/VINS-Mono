#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>

#include "camodocal/chessboard/Chessboard.h"
#include "camodocal/calib/CameraCalibration.h"
#include "camodocal/gpl/gpl.h"

int main(int argc, char** argv)
{
  cv::Size boardSize;
  float squareSize;
  std::string inputDir;
  std::string cameraModel;
  std::string cameraName;
  std::string prefix;
  std::string fileExtension;
  bool useOpenCV;
  bool viewResults;
  bool verbose;

  //========= Handling Program options =========
  boost::program_options::options_description desc("Allowed options"); // 创建一个options_description对象，用于存储所有可设置的选项，传入的参数是描述信息
  desc.add_options() // 添加选项，add_options()返回一个options_description_easy_init对象，该对象的operator()接受两个参数，第一个是选项名称，第二个是选项值的类型
    ("help", "produce help message")
    ("width, w", boost::program_options::value<int>(&boardSize.width)->default_value(8), "Number of inner corners on the chessboard pattern in x direction")
    ("height, h", boost::program_options::value<int>(&boardSize.height)->default_value(12), "Number of inner corners on the chessboard pattern in y direction")
    ("size, s", boost::program_options::value<float>(&squareSize)->default_value(7.f), "Size of one square in mm")
    ("input, i", boost::program_options::value<std::string>(&inputDir)->default_value("calibrationdata"), "Input directory containing chessboard images")
    ("prefix, p", boost::program_options::value<std::string>(&prefix)->default_value("left-"), "Prefix of images")
    ("file-extension, e", boost::program_options::value<std::string>(&fileExtension)->default_value(".png"), "File extension of images")
    ("camera-model", boost::program_options::value<std::string>(&cameraModel)->default_value("mei"), "Camera model: kannala-brandt | mei | pinhole")
    ("camera-name", boost::program_options::value<std::string>(&cameraName)->default_value("camera"), "Name of camera")
    ("opencv", boost::program_options::bool_switch(&useOpenCV)->default_value(true), "Use OpenCV to detect corners")
    ("view-results", boost::program_options::bool_switch(&viewResults)->default_value(false), "View results")
    ("verbose, v", boost::program_options::bool_switch(&verbose)->default_value(true), "Verbose output");
     // ::value<T>(&var)表示将选项值存储到var中，::default_value<T>(val)表示选项的默认值为val，bool_switch比较特殊，如果选项被出现，var为true，否则为false

  boost::program_options::positional_options_description pdesc; // 存储所有位置参数，以位置来识别参数，不需要额外的选项标志或名称
  pdesc.add("input", 1);

  boost::program_options::variables_map vm; // command_line_parser 结合位置描述来解析命令行参数，将解析结果存储到vm中
  boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
  boost::program_options::notify(vm); // 检查选项是否被设置，如果选项没有被设置，且没有默认值，则抛出异常

  if (vm.count("help"))
  {
    std::cout << desc << std::endl;
    return 1;
  }

  if (!boost::filesystem::exists(inputDir) && !boost::filesystem::is_directory(inputDir)) // 检查输入目录是否存在
  {
    std::cerr << "# ERROR: Cannot find input directory " << inputDir << "." << std::endl;
    return 1;
  }

  camodocal::Camera::ModelType modelType;
  if (boost::iequals(cameraModel, "kannala-brandt")) // 比较两个字符串是否相等，忽略大小写
  {
    modelType = camodocal::Camera::KANNALA_BRANDT;
  }
  else if (boost::iequals(cameraModel, "mei"))
  {
    modelType = camodocal::Camera::MEI;
  }
  else if (boost::iequals(cameraModel, "pinhole"))
  {
    modelType = camodocal::Camera::PINHOLE;
  }
  else if (boost::iequals(cameraModel, "scaramuzza"))
  {
    modelType = camodocal::Camera::SCARAMUZZA;
  }
  else
  {
    std::cerr << "# ERROR: Unknown camera model: " << cameraModel << std::endl;
    return 1;
  }

  switch (modelType)
  {
  case camodocal::Camera::KANNALA_BRANDT:
    std::cout << "# INFO: Camera model: Kannala-Brandt" << std::endl;
    break;
  case camodocal::Camera::MEI:
    std::cout << "# INFO: Camera model: Mei" << std::endl;
    break;
  case camodocal::Camera::PINHOLE:
    std::cout << "# INFO: Camera model: Pinhole" << std::endl;
    break;
  case camodocal::Camera::SCARAMUZZA:
    std::cout << "# INFO: Camera model: Scaramuzza-Omnidirect" << std::endl;
    break;
  }

  // look for images in input directory
  std::vector<std::string> imageFilenames; // boost::filesystem::directory_iterator用于遍历目录中的所有条目（只可以访问直接条目）的迭代器
  for (boost::filesystem::directory_iterator itr(inputDir); itr != boost::filesystem::directory_iterator(); ++itr)
  { // 确定给定的文件系统条目是否是一个常规文件，而不是目录、符号链接、块设备等其他类型的文件系统条目
    if (!boost::filesystem::is_regular_file(itr->status())) // status()返回文件状态信息
    {
      continue;
    }

    std::string filename = itr->path().filename().string(); // path()返回完整的文件路径，filename()返回文件名

    // check if prefix matches
    if (!prefix.empty())
    {
      if (filename.compare(0, prefix.length(), prefix) != 0)
      {
        continue;
      }
    }

    // check if file extension matches
    if (filename.compare(filename.length() - fileExtension.length(), fileExtension.length(), fileExtension) != 0)
    {
      continue;
    }

    imageFilenames.push_back(itr->path().string()); // 把符合条件的图片路径存储到imageFilenames中

    if (verbose)
    {
      std::cerr << "# INFO: Adding " << imageFilenames.back() << std::endl;
    }
  }

  if (imageFilenames.empty())
  {
    std::cerr << "# ERROR: No chessboard images found." << std::endl;
    return 1;
  }

  if (verbose)
  {
    std::cerr << "# INFO: # images: " << imageFilenames.size() << std::endl;
  }

  std::sort(imageFilenames.begin(), imageFilenames.end());

  cv::Mat image = cv::imread(imageFilenames.front(), -1); // -1 表示图像应该以其原始深度和通道数进行加载
  const cv::Size frameSize = image.size();

  camodocal::CameraCalibration calibration(modelType, cameraName, frameSize, boardSize, squareSize);
  calibration.setVerbose(verbose);

  std::vector<bool> chessboardFound(imageFilenames.size(), false);
  for (size_t i = 0; i < imageFilenames.size(); ++i)
  {
    image = cv::imread(imageFilenames.at(i), -1);

    camodocal::Chessboard chessboard(boardSize, image);

    chessboard.findCorners(useOpenCV);
    if (chessboard.cornersFound())
    {
      if (verbose)
      {
        std::cerr << "# INFO: Detected chessboard in image " << i + 1 << ", " << imageFilenames.at(i) << std::endl;
      }

      calibration.addChessboardData(chessboard.getCorners());

      cv::Mat sketch;
      chessboard.getSketch().copyTo(sketch);

      cv::imshow("Image", sketch);
      cv::waitKey(50);
    }
    else if (verbose)
    {
      std::cerr << "# INFO: Did not detect chessboard in image " << i + 1 << std::endl;
    }
    chessboardFound.at(i) = chessboard.cornersFound();
  }
  cv::destroyWindow("Image");

  if (calibration.sampleCount() < 10)
  {
    std::cerr << "# ERROR: Insufficient number of detected chessboards." << std::endl;
    return 1;
  }

  if (verbose)
  {
    std::cerr << "# INFO: Calibrating..." << std::endl;
  }

  double startTime = camodocal::timeInSeconds();

  calibration.calibrate();
  calibration.writeParams(cameraName + "_camera_calib.yaml");
  calibration.writeChessboardData(cameraName + "_chessboard_data.dat");

  if (verbose)
  {
    std::cout << "# INFO: Calibration took a total time of "
              << std::fixed << std::setprecision(3) << camodocal::timeInSeconds() - startTime
              << " sec.\n";
  }

  if (verbose)
  {
    std::cerr << "# INFO: Wrote calibration file to " << cameraName + "_camera_calib.yaml" << std::endl;
  }

  if (viewResults)
  {
    std::vector<cv::Mat> cbImages;
    std::vector<std::string> cbImageFilenames;

    for (size_t i = 0; i < imageFilenames.size(); ++i)
    {
      if (!chessboardFound.at(i))
      {
        continue;
      }

      cbImages.push_back(cv::imread(imageFilenames.at(i), -1));
      cbImageFilenames.push_back(imageFilenames.at(i));
    }

    // visualize observed and reprojected points
    calibration.drawResults(cbImages);

    for (size_t i = 0; i < cbImages.size(); ++i)
    {
<<<<<<< HEAD
      cv::putText(cbImages.at(i), cbImageFilenames.at(i), cv::Point(10,20),
                  cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255),
                  1, CV_AA);
      cv::imshow("Image", cbImages.at(i));
      cv::waitKey(0);
=======
        std::vector<cv::Mat> cbImages;
        std::vector<std::string> cbImageFilenames;

        for (size_t i = 0; i < imageFilenames.size(); ++i)
        {
            if (!chessboardFound.at(i))
            {
                continue;
            }

            cbImages.push_back(cv::imread(imageFilenames.at(i), -1));
            cbImageFilenames.push_back(imageFilenames.at(i));
        }

        // visualize observed and reprojected points
        calibration.drawResults(cbImages);

        for (size_t i = 0; i < cbImages.size(); ++i)
        {
            cv::putText(cbImages.at(i), cbImageFilenames.at(i), cv::Point(10,20),
                        cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255),
                        1, cv::LINE_AA);
            cv::imshow("Image", cbImages.at(i));
            cv::waitKey(0);
        }
>>>>>>> 90dabb5ec79946ae42fd2e1e91d4e69aabe1e25d
    }
  }

  return 0;
}
