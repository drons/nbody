#ifndef NBTYPE_QUAD_H
#define NBTYPE_QUAD_H

#include <quadmath.h>
#include <QDebug>
#include <QTextStream>

inline __float128 sqrt(__float128 x)
{
	return sqrtq(x);
}

inline __float128 fabs(__float128 x)
{
	return fabsq(x);
}

inline QDebug operator << (QDebug g, __float128 v)
{
	g << static_cast<double>(v);
	return g;
}

inline QTextStream& operator << (QTextStream& g, __float128 v)
{
	g << static_cast<double>(v);
	return g;
}

#endif //NBTYPE_QUAD_H
