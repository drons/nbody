#ifndef NBODY_FCOMPUTE_BLOCK_H
#define NBODY_FCOMPUTE_BLOCK_H

#include "nbody_fcompute.h"

class nbody_fcompute_block : public nbody_fcompute
{
public:
	nbody_fcompute_block();
	void fcompute( const nbody_data* data, const nbvertex_t* vertites, nbvertex_t* dv );
};

#endif // NBODY_FCOMPUTE_BLOCK_H
