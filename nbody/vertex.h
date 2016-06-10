#ifndef _VERTEX_
#define _VERTEX_
#include <math.h>
#include <stddef.h>

template<class T>
struct vertex3
{
	typedef size_t		size_type;
	typedef T			value_type;
	T x,y,z;
    vertex3(void){x = y = z = 0;}
	vertex3(T value){x = y = z = value;}
	vertex3(int value){x = y = z = (T)value;}
	template<class V>
    vertex3(const V& copy) : x((T)copy.x),y((T)copy.y),z((T)copy.z){}
	template<class V>
    vertex3& operator = (const V& copy){ x = (T)copy.x; y = (T)copy.y; z = (T)copy.z; return *this;}
    vertex3(const T &x_,const  T &y_,const  T &z_) :	x(x_),y(y_),z(z_){}

    T*	data(void) {return &x;}
    const T*	data(void) const {return &x;}
	const T& operator [] ( int n ) const
	{
		return data()[n];
	}
	T& operator [] ( int n )
	{
		return data()[n];
	}
	const T& operator [] ( size_t n ) const
	{
		return data()[n];
	}
	T& operator [] ( size_t n )
	{
		return data()[n];
	}
    bool operator == (const vertex3<T>& V) const
	{
		return ( x == V.x && y == V.y && z == V.z );
	}
    bool operator != (const vertex3<T>& V) const
	{
		return ( x != V.x || y != V.y || z != V.z );
	}

    //************************************************************************************************
	// Vector Sum
	//************************************************************************************************
 	
    vertex3<T> operator + (const vertex3<T>& V) const
	{
        return vertex3<T>(x + V.x,y + V.y,z + V.z);
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
        return vertex3<T>(x - V.x,y - V.y,z - V.z);
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
        return vertex3<T>(x*a,y*a,z*a);
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
        return vertex3<T>(x/a,y/a,z/a);
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
    vertex3<T> operator - (void) const
	{
        return vertex3<T>(-x,-y,-z);
	}
	//************************************************************************************************
	//Scalar mul
	//************************************************************************************************
    T operator * (const vertex3<T>& V) const
	{
		return (x*V.x +y*V.y + z*V.z);
	}
	//************************************************************************************************
	//Vector mul
	//************************************************************************************************
    vertex3<T> operator ^ (const vertex3<T>& V) const
	{
        return vertex3<T>(	y*V.z - z*V.y,
						z*V.x - x*V.z,
						x*V.y - y*V.x	);
	}
	//************************************************************************************************
	//	Helpers
	//************************************************************************************************
    void normalize(void)
	{
		(*this) = (*this)/length();
	}
    T distance(const vertex3<T>& Vd) const
	{
        vertex3<T> V = (*this) - Vd;
		return V.length();
	}
    T norm(void) const
	{
		return ( x*x + y*y + z*z );
	}
    T length(void) const
	{
		return sqrt((*this)*(*this));
	}
    vertex3<T> mirror(const vertex3<T>& V) const
	{
		return (V + ((*this) - V)*((T)2));
	}
};

template<class T>
struct vertex4
{
	typedef size_t		size_type;
	typedef T			value_type;
	T x,y,z,w;
    vertex4(void){x = y = z = w = 0;}
	vertex4(T value){x = y = z = w = value;}
	vertex4(int value){x = y = z = w = (T)value;}
	template<class V>
    vertex4(const V& copy) : x((T)copy.x),y((T)copy.y),z((T)copy.z),w((T)copy.w){}
	template<class V>
    vertex4& operator = (const V& copy){ x = (T)copy.x; y = (T)copy.y; z = (T)copy.z; w = (T)copy.w; return *this;}
    vertex4(const T &x_,const  T &y_,const  T &z_,const  T &w_) :	x(x_),y(y_),z(z_),w(w_){}
};

#endif//_VERTEX_
