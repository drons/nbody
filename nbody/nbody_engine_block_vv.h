#ifndef NBODY_ENGINE_BLOCK_VV_H
#define NBODY_ENGINE_BLOCK_VV_H

#include "nbody_engine_openmp.h"

class nbody_engine_block_vv : public nbody_engine_openmp
{
public:
	nbody_engine_block_vv();
	virtual const char* type_name() const;
	virtual void fcompute( const nbcoord_t& t, const memory* y, memory* f, size_t yoff, size_t foff );
};

#endif // NBODY_ENGINE_BLOCK_VV_H
