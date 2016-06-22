#ifndef NBODY_ENGINE_SIMPLE_H
#define NBODY_ENGINE_SIMPLE_H

#include "nbody_engine.h"

class nbody_engine_simple : public nbody_engine
{
public:
	nbody_engine_simple();
	void fcompute( const nbody_data* data, const nbvertex_t* vertites, nbvertex_t* dv );
};

#endif // NBODY_ENGINE_SIMPLE_H
