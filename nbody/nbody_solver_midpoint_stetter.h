#ifndef NBODY_SOLVER_MIDPOINT_STETTER_H
#define NBODY_SOLVER_MIDPOINT_STETTER_H

#include "nbody_solver.h"

/*!
	Implicit midpoint method

	@see (9.15) at [1] p. 228
	[1] E. Hairer, S. P. Nørsett, G. Wanner
		Solving Ordinary Differential Equations I Nonstiff Problems
		Second Revised Edition, DOI 10.1007/978-3-540-78862-1, 2008
		http://www.hds.bme.hu/~fhegedus/00%20-%20Numerics/B1993%20Solving%20Ordinary%20Differential%20Equations%20I%20-%20Nonstiff%20Problems.pdf
*/
class NBODY_DLL nbody_solver_midpoint_stetter : public nbody_solver
{
	nbody_engine::memory_array	m_uv;
	nbody_engine::memory_array	m_fu;
	nbody_engine::memory*		m_tmp;
	nbody_engine::memory*		m_du;
public:
	nbody_solver_midpoint_stetter();
	~nbody_solver_midpoint_stetter();
	const char* type_name() const override;
	void advise(nbcoord_t dt) override;
};

#endif // NBODY_SOLVER_MIDPOINT_STETTER_H
