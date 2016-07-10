#ifndef NBODY_ENGINE_OPENCL_H
#define NBODY_ENGINE_OPENCL_H

#include "nbody_engine.h"

class nbody_engine_opencl : public nbody_engine
{
	class	smemory;
	struct	data;
	data*	d;
public:
	nbody_engine_opencl();
	~nbody_engine_opencl();
	virtual const char* type_name() const;
	virtual void init( nbody_data* data );
	virtual void get_data( nbody_data* data );
	virtual size_t problem_size() const;
	virtual memory* y();
	virtual void advise_time( const nbcoord_t& dt );
	virtual nbcoord_t get_time() const;
	virtual size_t get_step() const;

	virtual void fcompute( const nbcoord_t& t, const memory* y, memory* f, size_t yoff, size_t foff );

	virtual memory* malloc( size_t );
	virtual void free( memory* );
	virtual void memcpy( void* src, memory* dst );
	virtual void memcpy( memory*, void* );
	virtual void memcpy( memory* a, const memory* b, size_t aoff, size_t boff );

	virtual void fmadd( memory* a, const memory* b, const nbcoord_t& c );
	virtual void fmadd( memory* a, const memory* b, const memory* c, const nbcoord_t& d, size_t aoff, size_t boff, size_t coff );
	virtual void fmaddn( memory* a, const memory* b, const memory* c, size_t bstride, size_t aoff, size_t boff, size_t csize );
	virtual void fmaddn( memory* a, const memory* b, const memory* c, const memory* d, size_t cstride, size_t aoff, size_t boff, size_t coff, size_t dsize );
	virtual void fmaxabs( const memory* a, nbcoord_t& result );

	static int info();
};

#endif // NBODY_ENGINE_OPENCL_H
