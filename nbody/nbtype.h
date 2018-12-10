#ifndef NBTYPE_H
#define NBTYPE_H

#include "vertex.h"

#ifdef NB_COORD_TYPE
typedef NB_COORD_TYPE			nbcoord_t;
#else
typedef double					nbcoord_t;
#endif
typedef vertex3<nbcoord_t>		nbvertex_t;
typedef vertex4<float>			nbcolor_t;
typedef vertex3<float>			nb3f_t;
typedef vertex3<double>			nb3d_t;
typedef vertex4<float>			nb4f_t;
typedef vertex4<double>			nb4d_t;

#define NBODY_DATA_BLOCK_SIZE	64
#define NBODY_MIN_R				1e-8

#endif // NBTYPE_H
