#pragma once

#include <ctime>
#include <cstdlib>
#include <chrono>

class TicToc
{
 public:
   TicToc()
   {
     tic();
   }

   void tic()
   {
     start = std::chrono::system_clock::now();
   }

   double toc()
   {
     end = std::chrono::system_clock::now();
     // 第一个参数_Rep表示用什么类型的数据表示这个时间长度，第二个参数_Period表示时间的单位，如果第二个参数省略，则默认以秒为单位
     std::chrono::duration<double> elapsed_seconds = end - start;
     return elapsed_seconds.count() * 1000; // count() 方法返回存储在 duration 对象中的时间长度
   }

 private:
   std::chrono::time_point<std::chrono::system_clock> start, end;
};
