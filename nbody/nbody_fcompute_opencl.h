#ifndef NBODY_FCOMPUTE_OPENCL_H
#define NBODY_FCOMPUTE_OPENCL_H

#include "nbody_fcompute.h"

class nbody_fcompute_opencl : public nbody_fcompute
{
	struct	data;
	data*	d;
public:
	nbody_fcompute_opencl();
	~nbody_fcompute_opencl();
	virtual void fcompute( const nbody_data* data, const nbvertex_t* vertites, nbvertex_t* dv );
	static int info();
};

#endif // NBODY_FCOMPUTE_OPENCL_H
