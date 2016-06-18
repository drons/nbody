#ifndef NBODY_SOLVER_H
#define NBODY_SOLVER_H

#include "nbody_data.h"
#include "nbody_fcompute.h"

class nbody_solver
{
	nbody_data*							m_data;
	nbody_fcompute*						m_engine;
	nbcoord_t							m_min_step;
	nbcoord_t							m_max_step;
public:
	nbody_solver( nbody_data* data );
	virtual ~nbody_solver();
	nbody_data* data() const;
	void set_engine( nbody_fcompute* );
	nbody_fcompute* get_engine();
	void step_v( const nbvertex_t* vertites, nbvertex_t* dv );
	virtual void step( nbcoord_t dt ) = 0;
	void set_time_step( nbcoord_t min_step, nbcoord_t max_step );
	nbcoord_t get_min_step() const;
	nbcoord_t get_max_step() const;
	int run( nbcoord_t max_time, nbcoord_t dump_dt, nbcoord_t check_dt);
};

#endif // NBODY_SOLVER_H
