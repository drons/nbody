#ifndef NBODY_SOLVER_H
#define NBODY_SOLVER_H

#include "nbody_data.h"
#include "nbody_fcompute.h"

class nbody_solver
{
	nbody_data*							m_data;
	nbody_fcompute*						m_engine;
public:
	nbody_solver( nbody_data* data );
	virtual ~nbody_solver();
	nbody_data* data() const;
	void set_engine( nbody_fcompute* );
	void step_v( const nbvertex_t* vertites, nbvertex_t* dv );
	virtual void step( nbcoord_t dt ) = 0;
};

#endif // NBODY_SOLVER_H
