#ifndef NBTYPE_H
#define NBTYPE_H

#include "vertex.h"
#include <QGL>

typedef double					nbcoord_t;
typedef vertex3<nbcoord_t>		nbvertex_t;
typedef vertex3<float>			nb3f_t;
typedef vertex3<double>			nb3d_t;

template<class T>
class nbtype_info
{
private:
	nbtype_info();
	// Empty implementation
};

#define DECLARE_NBTYPE_INFO( TYPE, s, t, n )	\
template<>										\
class nbtype_info<TYPE>							\
{												\
public:											\
	nbtype_info(){}								\
	static GLint size() { return s;}			\
	static GLenum gl_type() { return t;}		\
	static const char* type_name() { return n;}	\
}

DECLARE_NBTYPE_INFO( float,				1, GL_FLOAT,	"float" );
DECLARE_NBTYPE_INFO( double,			1, GL_DOUBLE,	"double" );
DECLARE_NBTYPE_INFO( vertex3<float>,	3, GL_FLOAT,	"float3" );
DECLARE_NBTYPE_INFO( vertex3<double>,	3, GL_DOUBLE,	"double3" );

#endif // NBTYPE_H
