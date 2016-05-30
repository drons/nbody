#ifndef NBODY_SOLVER_ADAMS_H
#define NBODY_SOLVER_ADAMS_H

#include "nbody_solver.h"

class nbody_solver_adams : public nbody_solver
{
	nbody_solver*								m_starter;
	std::vector< nbvertex_t >					m_correction_vert;
	std::vector< nbvertex_t >					m_correction_vel;
	std::vector< std::vector< nbvertex_t > >	m_xdata;
	std::vector< std::vector< nbvertex_t > >	m_vdata;
	std::vector< nbvertex_t* >					m_dx;
	std::vector< nbvertex_t* >					m_dv;
public:
	nbody_solver_adams( nbody_data* data );
	~nbody_solver_adams();
	void step(double dt);
};

#endif // NBODY_SOLVER_ADAMS_H
