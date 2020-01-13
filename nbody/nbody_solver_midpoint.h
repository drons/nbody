#ifndef NBODY_SOLVER_MIDPOINT_H
#define NBODY_SOLVER_MIDPOINT_H

#include "nbody_solver.h"

/*!
	Midpoint method

	@see (17.1.2) at [1]
	[1] Numerical Recipes, Third Edition, Cambridge, 2007
*/
class NBODY_DLL nbody_solver_midpoint : public nbody_solver
{
	nbody_engine::memory*	m_k1;
	nbody_engine::memory*	m_k2;
	nbody_engine::memory*	m_tmp;
public:
	nbody_solver_midpoint();
	~nbody_solver_midpoint();
	const char* type_name() const override;
	void advise(nbcoord_t dt) override;
};

#endif // NBODY_SOLVER_MIDPOINT_H
