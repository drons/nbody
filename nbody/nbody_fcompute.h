#ifndef NBODY_FCOMPUTE_H
#define NBODY_FCOMPUTE_H

#include "nbody_data.h"

class nbody_fcompute
{
public:
	nbody_fcompute();
	virtual ~nbody_fcompute();
	virtual void fcompute( const nbody_data* data, const nbvertex_t* vertites, nbvertex_t* dv ) = 0;
};

#endif // NBODY_FCOMPUTE_H
