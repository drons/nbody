#ifndef NBODY_SOLVER_RUNGE_KUTTA_H
#define NBODY_SOLVER_RUNGE_KUTTA_H

#include "nbody_solver.h"

class nbody_solver_runge_kutta : public nbody_solver
{
	std::vector< nbvertex_t >	k1,k2,k3,k4;
	std::vector< nbvertex_t >	q1,q2,q3,q4;
	std::vector< nbvertex_t >	tmp;
public:
	nbody_solver_runge_kutta( nbody_data* data );
	virtual void step( double dt );
};

#endif // NBODY_SOLVER_RUNGE_KUTTA_H
