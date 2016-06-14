#ifndef NBODY_SOLVER_RKDP_H
#define NBODY_SOLVER_RKDP_H

#include "nbody_solver.h"

class nbody_solver_rkdp : public nbody_solver
{
	static const size_t			MAX_RECURSION = 8;
	std::vector< nbvertex_t >	k1,k2,k3,k4,k5,k6,k7;
	std::vector< nbvertex_t >	q1,q2,q3,q4,q5,q6,q7;
	std::vector< nbvertex_t >	tmpvrt;
	std::vector< nbvertex_t >	tmpvel;
	std::vector< nbvertex_t >	vertites_stack[MAX_RECURSION];
	std::vector< nbvertex_t >	velosites_stack[MAX_RECURSION];
	int							m_step_subdivisions;
	nbcoord_t					m_vrt_error_threshold;
	nbcoord_t					m_vel_error_threshold;
public:
	nbody_solver_rkdp( nbody_data* data );
	virtual void step( double dt );
private:
	void sub_step( size_t substeps_count, nbcoord_t dt, nbvertex_t* vertites, nbvertex_t* velosites, size_t recursion_level );
};

#endif // NBODY_SOLVER_RKDP_H
