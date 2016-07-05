#ifndef NBODY_ENGINE_BLOCK_H
#define NBODY_ENGINE_BLOCK_H

#include "nbody_engine_simple.h"

class nbody_engine_block : public nbody_engine_simple
{
public:
	nbody_engine_block();
	virtual void fcompute( const nbcoord_t& t, const memory* y, memory* f, size_t yoff, size_t foff );
};

#endif // NBODY_ENGINE_BLOCK_H
