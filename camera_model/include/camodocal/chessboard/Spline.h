/*  dynamo:- Event driven molecular dynamics simulator
    http://www.marcusbannerman.co.uk/dynamo
    Copyright (C) 2011  Marcus N Campbell Bannerman <m.bannerman@gmail.com>

    This program is free software: you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    version 3 as published by the Free Software Foundation.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <exception>

namespace ublas = boost::numeric::ublas;

// 私有继承的特点：1.访问控制方面：派生类对外时，基类原来的公有和保护访问控制权限会变为私有；2.接口不可访问：使用私有继承，基类的共有成员方法在派生类外不可访问；
// 3.子类对象不可隐式地转换为基类对象
class Spline : private std::vector<std::pair<double, double>> {
 public:
   //The boundary conditions available
   enum BC_type {
	 FIXED_1ST_DERIV_BC,  //The first derivative at the end points is fixed
	 FIXED_2ND_DERIV_BC,  //The second derivative at the end points is fixed
	 PARABOLIC_RUNOUT_BC  //M0 = M1 and Mn-1 = Mn，即端点处的二阶导数相等
   };

   enum Spline_type {
	 LINEAR,
	 CUBIC
   };

  //Constructor takes the boundary conditions as arguments, this
  //sets the first derivative (gradient) at the lower and upper
  //end points
  Spline():
	_valid(false),
	_BCLow(FIXED_2ND_DERIV_BC), _BCHigh(FIXED_2ND_DERIV_BC),
	_BCLowVal(0), _BCHighVal(0),
	_type(CUBIC)
  {}

  typedef std::vector<std::pair<double, double>> base; // base是vector的别名
  typedef base::const_iterator const_iterator;

  //Standard STL read-only container stuff
  const_iterator begin() const { return base::begin(); }
  const_iterator end() const { return base::end(); }
  void clear() { _valid = false; base::clear(); _data.clear(); }
  size_t size() const { return base::size(); }
  size_t max_size() const { return base::max_size(); }
  size_t capacity() const { return base::capacity(); }
  bool empty() const { return base::empty(); }

  //Add a point to the spline, and invalidate it so it's
  //recalculated on the next access 将一个新的采样点加入到样条函数中
  inline void addPoint(double x, double y)
  {
	_valid = false;
	base::push_back(std::pair<double, double>(x,y)); // 构造派生类的对象时，先调用基类的构造函数，析构的顺序相反，派生类包含了从基类继承来的数据成员
  }// 和它自己的数据成员，一个派生类对象通常在内存中表示为一个连续的内存块。这个内存块首先包含从基类继承来的数据成员，然后是派生类自己的数据成员

  //Reset the boundary conditions
  inline void setLowBC(BC_type BC, double val = 0)
  { _BCLow = BC; _BCLowVal = val; _valid = false; }

  inline void setHighBC(BC_type BC, double val = 0)
  { _BCHigh = BC; _BCHighVal = val; _valid = false; }

  void setType(Spline_type type) { _type = type; _valid = false; } // 设置样条函数的类型

  //Check if the spline has been calculated, then generate the
  //spline interpolated value
  double operator()(double xval)
  {
	if (!_valid) generate(); // 如果还没有计算过样条参数，就计算一下

	//Special cases when we're outside the range of the spline points
	if (xval <= x(0)) return lowCalc(xval); // 小于第一个采样点的x值，就返回下边界处的值
	if (xval >= x(size()-1)) return highCalc(xval); // 大于最后一个采样点的x值，就返回上边界处的值

	//Check all intervals except the last one
	for (std::vector<SplineData>::const_iterator iPtr = _data.begin();
		 iPtr != _data.end()-1; ++iPtr)
		if ((xval >= iPtr->x) && (xval <= (iPtr+1)->x))
		  return splineCalc(iPtr, xval);

	return splineCalc(_data.end() - 1, xval);
  }

  std::vector<double> getparam(int i) {
    std::vector<double> param;
    param.push_back(_data[i].x);
    param.push_back(_data[i].a);
    param.push_back(_data[i].b);
    param.push_back(_data[i].c);
    param.push_back(_data[i].d);
    return param;
  }

private:

  ///////PRIVATE DATA MEMBERS
  struct SplineData { double x,a,b,c,d; };
  //vector of calculated spline data
  std::vector<SplineData> _data; // 要计算样条函数值的x值和x所在样条分段上的参数
  //Second derivative at each point
  ublas::vector<double> _ddy;  // boost的向量库，和std::vector类似
  //Tracks whether the spline parameters have been calculated for
  //the current set of points
  bool _valid; // 是否已经计算过本组数据的样条参数
  //The boundary conditions
  BC_type _BCLow, _BCHigh; // 边界条件的类型
  //The values of the boundary conditions
  double _BCLowVal, _BCHighVal; // 边界上一阶导数或者二阶导数的值

  Spline_type _type; //线性还是三次样条

  ///////PRIVATE FUNCTIONS
  //Function to calculate the value of a given spline at a point xval，SplineData里的x是每一段的起始点
  inline double splineCalc(std::vector<SplineData>::const_iterator i, double xval)
  {
	const double lx = xval - i->x;
	return ((i->a * lx + i->b) * lx + i->c) * lx + i->d;
  }

  inline double lowCalc(double xval)
  {
	const double lx = xval - x(0);

	if (_type == LINEAR)
	  return lx * _BCHighVal + y(0);

	const double firstDeriv = (y(1) - y(0)) / h(0) - 2 * h(0) * (_data[0].b + 2 * _data[1].b) / 6;

	switch(_BCLow)
	  {
	  case FIXED_1ST_DERIV_BC:
		return lx * _BCLowVal + y(0);
	  case FIXED_2ND_DERIV_BC:
		  return lx * lx * _BCLowVal + firstDeriv * lx + y(0);
	  case PARABOLIC_RUNOUT_BC:
		return lx * lx * _ddy[0] + lx * firstDeriv  + y(0);
	  }
	throw std::runtime_error("Unknown BC");
  }

  inline double highCalc(double xval)
  {
	const double lx = xval - x(size() - 1);

	if (_type == LINEAR)
	  return lx * _BCHighVal + y(size() - 1); // 如果是线性样条，那么_BCHighVal就是上边界处的斜率
    // 根据3次样条的公式，带入计算边界处的一阶导数
	const double firstDeriv = h(size() - 2) * (_ddy[size() - 2] + 2 * _ddy[size() - 1]) / 6 + (y(size() - 1) - y(size() - 2)) / h(size() - 2);

	switch(_BCHigh)
	  {
	  case FIXED_1ST_DERIV_BC:
		return lx * _BCHighVal + y(size() - 1); // 如果是固定一阶导数的边界条件，那么使用线性外推
	  case FIXED_2ND_DERIV_BC:
		return 0.5 * lx * lx * _BCHighVal + firstDeriv * lx + y(size() - 1); // 如果是固定二阶导数的边界条件，那么使用二次外推(泰勒展开到二阶)
	  case PARABOLIC_RUNOUT_BC:
		return lx * lx * _ddy[size()-1] + lx * firstDeriv  + y(size() - 1);
	  }
	throw std::runtime_error("Unknown BC");
  }

  // These just provide access to the point data in a clean way
  inline double x(size_t i) const { return operator[](i).first; }  // operator[]是vector的成员函数，因为Spline继承了vector
  inline double y(size_t i) const { return operator[](i).second; } // x, y函数返回第i个采样点的x, y坐标
  inline double h(size_t i) const { return x(i + 1) - x(i); }

  //Invert a arbitrary matrix using the boost ublas library
  template<class T>
  bool InvertMatrix(ublas::matrix<T> A,
		ublas::matrix<T>& inverse)
  {
	using namespace ublas;

	// create a permutation matrix for the LU-factorization
	permutation_matrix<std::size_t> pm(A.size1());

	// perform LU-factorization
	int res = lu_factorize(A,pm);
		if( res != 0 ) return false;

	// create identity matrix of "inverse"
	inverse.assign(ublas::identity_matrix<T>(A.size1()));

	// backsubstitute to get the inverse
	lu_substitute(A, pm, inverse);

	return true;
  }

  //This function will recalculate the spline parameters and store
  //them in _data, ready for spline interpolation
  void generate()
  {
	if (size() < 2)
	  throw std::runtime_error("Spline requires at least 2 points"); // 如果少于两个点，就抛出异常，并打印括号里的错误提示

	//If any spline points are at the same x location, we have to
	//just slightly separate them 如果任何样条点位于相同的 x 位置，我们必须稍微分开它们
	{
	  bool testPassed(false);
	  while (!testPassed)
      {
        testPassed = true;
        std::sort(base::begin(), base::end()); // 对vector里的pair按照first排序（即按照x排序）

        for (auto iPtr = base::begin(); iPtr != base::end() - 1; ++iPtr) {
          if (iPtr->first == (iPtr + 1)->first) {
            if ((iPtr + 1)->first != 0)
              (iPtr + 1)->first += (iPtr + 1)->first * std::numeric_limits<double>::epsilon() * 10;
            else
              (iPtr + 1)->first = std::numeric_limits<double>::epsilon() * 10;
            testPassed = false;
            break;
          }
        }
      }
	}

	const size_t e = size() - 1; // 样条分段的个数

	switch (_type)
    {
    case LINEAR:
    {
      _data.resize(e); // 存储每一段的参数，并重新分配内存空间
      for (size_t i(0); i < e; ++i)
      {
        _data[i].x = x(i); // 每一段的起始点
        _data[i].a = 0; // 每一段3次项前面的系数
        _data[i].b = 0;
        _data[i].c = (y(i+1) - y(i)) / (x(i+1) - x(i));
        _data[i].d = y(i);
      }
      break;
    }
    case CUBIC:
    {
      ublas::matrix<double> A(size(), size());
      for (size_t yv(0); yv <= e; ++yv)
      for (size_t xv(0); xv <= e; ++xv)
        A(xv,yv) = 0; // 将A矩阵初始化为0

      for (size_t i(1); i < e; ++i) // 未知数M是按行排列
      {
        A(i - 1,i) = h(i - 1);
        A(i,i) = 2 * (h(i - 1) + h(i));
        A(i + 1,i) = h(i);
      }

      ublas::vector<double> C(size());
      for (size_t xv(0); xv <= e; ++xv)
        C(xv) = 0;

      for (size_t i(1); i < e; ++i)
        C(i) = 6 * ((y(i + 1) - y(i)) / h(i) - (y(i) - y(i - 1)) / h(i-1));

      //Boundary conditions
      switch(_BCLow)
    {
    case FIXED_1ST_DERIV_BC:
      C(0) = 6 * ((y(1) - y(0)) / h(0) - _BCLowVal);
      A(0,0) = 2 * h(0);
      A(1,0) = h(0);
      break;
    case FIXED_2ND_DERIV_BC:
      C(0) = _BCLowVal;
      A(0,0) = 1;
      break;
    case PARABOLIC_RUNOUT_BC:
      C(0) = 0; A(0,0) = 1; A(1,0) = -1;
      break;
    }

      switch(_BCHigh)
    {
    case FIXED_1ST_DERIV_BC:
      C(e) = 6 * (_BCHighVal - (y(e) - y(e-1)) / h(e-1));
      A(e,e) = 2 * h(e - 1);
      A(e-1,e) = h(e - 1);
      break;
    case FIXED_2ND_DERIV_BC:
      C(e) = _BCHighVal;
      A(e,e) = 1;
      break;
    case PARABOLIC_RUNOUT_BC:
      C(e) = 0; A(e,e) = 1; A(e-1,e) = -1;
      break;
    }

      ublas::matrix<double> AInv(size(), size());
      InvertMatrix(A,AInv); // 求A的逆矩阵

      _ddy = ublas::prod(C, AInv); // 端点处的二阶导数

      _data.resize(size()-1);
      for (size_t i(0); i < e; ++i)
    {
      _data[i].x = x(i);
      _data[i].a = (_ddy(i+1) - _ddy(i)) / (6 * h(i));
      _data[i].b = _ddy(i) / 2;
      _data[i].c = (y(i+1) - y(i)) / h(i) - _ddy(i+1) * h(i) / 6 - _ddy(i) * h(i) / 3;
      _data[i].d = y(i);
    }
    }
  }
	_valid = true;
  }
};
