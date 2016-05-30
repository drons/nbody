#ifndef NBODY_SOLVER_H
#define NBODY_SOLVER_H

#include "nbody_data.h"

class nbody_solver
{
	nbody_data*							m_data;
	std::vector< std::vector<size_t> >	m_adjacent_body;
	std::vector< nbvertex_t >			m_univerce_force;
public:
	nbody_solver( nbody_data* data );
	virtual ~nbody_solver();
	nbody_data* data() const;
	void step_v( const nbvertex_t* vertites, nbvertex_t* dv );
	void step_v0( const nbvertex_t* vertites, nbvertex_t* dv );
	void step_v2( const nbvertex_t* vertites, nbvertex_t* dv );
	virtual void step( nbcoord_t dt ) = 0;
};

#endif // NBODY_SOLVER_H
