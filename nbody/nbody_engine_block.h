#ifndef NBODY_ENGINE_BLOCK_H
#define NBODY_ENGINE_BLOCK_H

#include "nbody_engine.h"

class nbody_engine_block : public nbody_engine
{
public:
	nbody_engine_block();
	void fcompute( const nbody_data* data, const nbvertex_t* vertites, nbvertex_t* dv );
};

#endif // NBODY_ENGINE_BLOCK_H
