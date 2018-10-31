#ifndef _VERTEX_
#define _VERTEX_
#include <math.h>
#include <stddef.h>

template<class T>
struct vertex3
{
	typedef size_t		size_type;
	typedef T			value_type;
	T x, y, z;
	vertex3() {x = y = z = 0;}
	explicit vertex3(T value) {x = y = z = value;}
	explicit vertex3(int value) {x = y = z = static_cast<T>(value);}
	template<class V>
	explicit vertex3(const V& copy) :
		x(static_cast<T>(copy.x)),
		y(static_cast<T>(copy.y)),
		z(static_cast<T>(copy.z)) {}
	template<class V>
	vertex3& operator = (const V& copy)
	{
		x = static_cast<T>(copy.x);
		y = static_cast<T>(copy.y);
		z = static_cast<T>(copy.z);
		return *this;
	}
	vertex3(const T& x_, const  T& y_, const  T& z_) :	x(x_), y(y_), z(z_) {}

	T*	data() {return &x;}
	const T*	data() const {return &x;}
	const T& operator [](size_type n) const
	{
		return data()[n];
	}
	T& operator [](size_type n)
	{
		return data()[n];
	}
	bool operator == (const vertex3<T>& V) const
	{
		return (x == V.x && y == V.y && z == V.z);
	}
	bool operator != (const vertex3<T>& V) const
	{
		return (x != V.x || y != V.y || z != V.z);
	}

	//************************************************************************************************
	// Vector Sum
	//************************************************************************************************
	vertex3<T> operator + (const vertex3<T>& V) const
	{
		return vertex3<T>(x + V.x, y + V.y, z + V.z);
	}
	vertex3<T>& operator += (const vertex3<T>& V)
	{
		x += V.x;
		y += V.y;
		z += V.z;
		return *this;
	}
	//************************************************************************************************
	//Vector Sub
	//************************************************************************************************
	vertex3<T> operator - (const vertex3<T>& V) const
	{
		return vertex3<T>(x - V.x, y - V.y, z - V.z);
	}
	vertex3<T>& operator -= (const vertex3<T>& V)
	{
		x -= V.x;
		y -= V.y;
		z -= V.z;
		return *this;
	}
	//************************************************************************************************
	//Mul
	//************************************************************************************************
	vertex3<T> operator * (const T a) const
	{
		return vertex3<T>(x * a, y * a, z * a);
	}
	vertex3<T>& operator *= (const T a)
	{
		x *= a;
		y *= a;
		z *= a;
		return *this;
	}
	//************************************************************************************************
	//Div
	//************************************************************************************************
	vertex3<T> operator / (const T a) const

	{
		return vertex3<T>(x / a, y / a, z / a);
	}
	vertex3<T>& operator /=(const T a)
	{
		x /= a;
		y /= a;
		z /= a;
		return *this;
	}
	vertex3<T>& operator /= (const vertex3<T>& V)
	{
		x /= V.x;
		y /= V.y;
		z /= V.z;
		return *this;
	}
	//************************************************************************************************
	//Unary minus
	//************************************************************************************************
	vertex3<T> operator - () const
	{
		return vertex3<T>(-x, -y, -z);
	}
	//************************************************************************************************
	//Scalar mul
	//************************************************************************************************
	T operator * (const vertex3<T>& V) const
	{
		return (x * V.x + y * V.y + z * V.z);
	}
	//************************************************************************************************
	//Vector mul
	//************************************************************************************************
	vertex3<T> operator ^ (const vertex3<T>& V) const
	{
		return vertex3<T>(y * V.z - z * V.y,
						  z * V.x - x * V.z,
						  x * V.y - y * V.x);
	}
	//************************************************************************************************
	//	Helpers
	//************************************************************************************************
	void normalize()
	{
		(*this) = (*this) / length();
	}
	T distance(const vertex3<T>& Vd) const
	{
		vertex3<T> V = (*this) - Vd;
		return V.length();
	}
	T norm() const
	{
		return (x * x + y * y + z * z);
	}
	T length() const
	{
		return sqrt((*this) * (*this));
	}
	vertex3<T> mirror(const vertex3<T>& V) const
	{
		return (V + ((*this) - V) * static_cast<T>(2));
	}
};

template<class T>
struct vertex4
{
	typedef size_t		size_type;
	typedef T			value_type;
	T x, y, z, w;
	vertex4() {x = y = z = w = 0;}
	explicit vertex4(T value) {x = y = z = w = value;}
	explicit vertex4(int value) {x = y = z = w = static_cast<T>(value);}
	template<class V>
	explicit vertex4(const V& copy) :
		x(static_cast<T>(copy.x)),
		y(static_cast<T>(copy.y)),
		z(static_cast<T>(copy.z)),
		w(static_cast<T>(copy.w))
	{}
	template<class V>
	vertex4& operator = (const V& copy)
	{
		x = static_cast<T>(copy.x);
		y = static_cast<T>(copy.y);
		z = static_cast<T>(copy.z);
		w = static_cast<T>(copy.w);
		return *this;
	}
	vertex4(const T& x_, const  T& y_, const  T& z_, const  T& w_) :
		x(x_), y(y_), z(z_), w(w_)
	{}

	bool operator == (const vertex4<T>& V) const
	{
		return (x == V.x && y == V.y && z == V.z && w == V.w);
	}
	//************************************************************************************************
	// Vector Sum
	//************************************************************************************************
	vertex4<T> operator + (const vertex4<T>& V) const
	{
		return vertex4<T>(x + V.x, y + V.y, z + V.z, w + V.w);
	}
	vertex4<T>& operator += (const vertex4<T>& V)
	{
		x += V.x;
		y += V.y;
		z += V.z;
		w += V.w;
		return *this;
	}

	//************************************************************************************************
	//Vector Sub
	//************************************************************************************************
	vertex4<T> operator - (const vertex4<T>& V) const
	{
		return vertex4<T>(x - V.x, y - V.y, z - V.z, w - V.w);
	}
	vertex4<T>& operator -= (const vertex4<T>& V)
	{
		x -= V.x;
		y -= V.y;
		z -= V.z;
		w -= V.w;
		return *this;
	}
};

#endif//_VERTEX_
