#ifndef NBTYPE_H
#define NBTYPE_H

#ifndef NB_COORD_PRECISION
#define NB_COORD_PRECISION 2
#endif

#if NB_COORD_PRECISION == 1
typedef float nbcoord_t;
#elif NB_COORD_PRECISION == 2
typedef double nbcoord_t;
#elif NB_COORD_PRECISION == 4
#include "nbtype_quad.h"
typedef __float128 nbcoord_t;
#endif

#include "vertex.h"

typedef vertex3<nbcoord_t>		nbvertex_t;
typedef vertex4<float>			nbcolor_t;
typedef vertex3<float>			nb3f_t;
typedef vertex3<double>			nb3d_t;
typedef vertex4<float>			nb4f_t;
typedef vertex4<double>			nb4d_t;

#define NBODY_DATA_BLOCK_SIZE	64
#define NBODY_MIN_R				1e-8

#endif // NBTYPE_H
