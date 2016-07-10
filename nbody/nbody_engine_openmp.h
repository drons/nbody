#ifndef NBODY_ENGINE_OPENMP_H
#define NBODY_ENGINE_OPENMP_H

#include "nbody_engine_simple.h"

class nbody_engine_openmp : public nbody_engine_simple
{
public:
	nbody_engine_openmp();
	~nbody_engine_openmp();

	virtual const char* type_name() const;

	virtual void fcompute( const nbcoord_t& t, const memory* y, memory* f, size_t yoff, size_t foff );

	virtual void memcpy( memory* a, const memory* b, size_t aoff, size_t boff );

	virtual void fmadd( memory* a, const memory* b, const nbcoord_t& c );
	virtual void fmadd( memory* a, const memory* b, const memory* c, const nbcoord_t& d, size_t aoff, size_t boff, size_t coff );
	virtual void fmaddn( memory* a, const memory* b, const memory* c, size_t bstride, size_t aoff, size_t boff, size_t csize );
	virtual void fmaddn( memory* a, const memory* b, const memory* c, const memory* d, size_t cstride, size_t aoff, size_t boff, size_t coff, size_t dsize );
	virtual void fmaxabs( const memory* a, nbcoord_t& result );
};

#endif // NBODY_ENGINE_SIMPLE_H
