#ifndef NBTYPE_QUAD_H
#define NBTYPE_QUAD_H

#include <quadmath.h>
#include <QDebug>
#include <QTextStream>
#include <limits>

namespace std {
template<>
struct numeric_limits<__float128>
{
	static constexpr bool is_specialized = true;
	static __float128 max()
	{
		return strtoflt128("1.18973149535723176508575932662800702e4932", nullptr);
	}
	static __float128 min()
	{
		return strtoflt128("1.18973149535723176508575932662800702e4932", nullptr);
	}
	static __float128 epsilon()
	{
		return strtoflt128("1.92592994438723585305597794258492732e-34", nullptr);
	}
};
}// namespace std

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
	char	buf[256] = {0};
	quadmath_snprintf(buf, sizeof(buf), "%+-#*.20Qe", 32, v);
	g << buf;
	return g;
}

inline QTextStream& operator << (QTextStream& g, __float128 v)
{
	char	buf[256] = {0};
	quadmath_snprintf(buf, sizeof(buf), "%+-#*.20Qe", g.realNumberPrecision(), v);
	g << buf;
	return g;
}

#endif //NBTYPE_QUAD_H
