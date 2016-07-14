#ifndef NBODY_ENGINE_SPARSE_H
#define NBODY_ENGINE_SPARSE_H

#include "nbody_engine.h"

class nbody_engine_sparse : public nbody_engine
{
	std::vector< std::vector<size_t> >	m_adjacent_body;
	std::vector< nbvertex_t >			m_univerce_force;
public:
	nbody_engine_sparse();
	void fcompute_sparce( const nbody_data* data, const nbvertex_t* vertites, nbvertex_t* dv );
};

#endif // NBODY_ENGINE_SPARSE_H
