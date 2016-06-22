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
	virtual void init( const nbody_data* data );
	virtual void fcompute( const nbcoord_t& t, const memory* y, memory* f );

	virtual memory* malloc( size_t );
	virtual void free( memory* );

	static int info();
};

#endif // NBODY_ENGINE_OPENCL_H
