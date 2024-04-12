#ifndef CHESSBOARDCORNER_H
#define CHESSBOARDCORNER_H

#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>

namespace camodocal
{

class ChessboardCorner;
typedef boost::shared_ptr<ChessboardCorner> ChessboardCornerPtr;

class ChessboardCorner
{
 public:
   ChessboardCorner() : row(0), column(0), needsNeighbor(true), count(0) {}

   float meanDist(int &n) const
   {
     float sum = 0;
     n = 0;
     for (const auto& neighbor : neighbors)
     {
       if (neighbor.get()) // neighbor.get()返回原始的指针
       {
         float dx = neighbor->pt.x - pt.x;
         float dy = neighbor->pt.y - pt.y;
         sum += std::sqrt(dx * dx + dy * dy);
         n++;
       }
     }
      return sum / (float)std::max(n, 1);
   }

   cv::Point2f pt;                     // X and y coordinates
   int row;                            // Row and column of the corner
   int column;                         // in the found pattern
   bool needsNeighbor;                 // Does the corner require a neighbor?
   int count;                          // number of corner neighbors
   ChessboardCornerPtr neighbors[4];   // pointer to all corner neighbors
};

}

#endif
