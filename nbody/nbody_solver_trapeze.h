#ifndef NBODY_SOLVER_TRAPEZE_H
#define NBODY_SOLVER_TRAPEZE_H

#include "nbody_solver.h"

class nbody_solver_trapeze : public nbody_solver
{
	std::vector< nbvertex_t >	m_dv0;
	std::vector< nbvertex_t >	m_dv1;
	std::vector< nbvertex_t >	m_predictor_vert;
	std::vector< nbvertex_t >	m_predictor_vel;
	std::vector< nbvertex_t >	m_velosites0;
public:
	nbody_solver_trapeze( nbody_data* data );
	virtual void step( nbcoord_t dt );
};

#endif // NBODY_SOLVER_TRAPEZE_H
