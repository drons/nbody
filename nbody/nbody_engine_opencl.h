#ifndef NBODY_ENGINE_OPENCL_H
#define NBODY_ENGINE_OPENCL_H

#include "nbody_engine.h"

class nbody_engine_opencl : public nbody_engine
{
	struct	data;
	data*	d;
public:
	nbody_engine_opencl();
	~nbody_engine_opencl();
	virtual void fcompute( const nbody_data* data, const nbvertex_t* vertites, nbvertex_t* dv );
	static int info();
};

#endif // NBODY_ENGINE_OPENCL_H
