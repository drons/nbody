#ifndef NBODY_SOLVER_STORMER_H
#define NBODY_SOLVER_STORMER_H

#include "nbody_solver.h"

class NBODY_DLL nbody_solver_stormer : public nbody_solver
{
	std::vector< nbvertex_t >	m_dv;
	std::vector< nbvertex_t >	m_prev_vert;
	std::vector< nbvertex_t >	m_correction_vert;
	std::vector< nbvertex_t >	m_correction_vel;
public:
	nbody_solver_stormer();
	const char* type_name() const override;
	void advise(nbcoord_t dt) override;
};

#endif // NBODY_SOLVER_STORMER_H
