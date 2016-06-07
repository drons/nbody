#ifndef NBODY_FCOMPUTE_SPARSE_H
#define NBODY_FCOMPUTE_SPARSE_H

#include "nbody_fcompute.h"

class nbody_fcompute_sparse : public nbody_fcompute
{
	std::vector< std::vector<size_t> >	m_adjacent_body;
	std::vector< nbvertex_t >			m_univerce_force;
public:
	nbody_fcompute_sparse();
	void fcompute( const nbody_data* data, const nbvertex_t* vertites, nbvertex_t* dv );
};

#endif // NBODY_FCOMPUTE_SPARSE_H
