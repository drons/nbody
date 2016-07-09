#ifndef NBODY_SOLVER_ADAMS_H
#define NBODY_SOLVER_ADAMS_H

#include "nbody_solver.h"

/*!
   \brief Adams–Bashforth/Moulton method
 */
class nbody_solver_adams : public nbody_solver
{
	nbody_solver*			m_starter;
	nbody_engine::memory*	m_f;
	nbody_engine::memory*	m_predictor;
	nbody_engine::memory*	m_coeff;
	size_t					m_rank;
	size_t					m_refine_steps_count;
	bool					m_implicit;
public:
	nbody_solver_adams( size_t rank = 5, bool implicit = false );
	~nbody_solver_adams();
	const char* type_name() const;
	void set_refine_steps_count( size_t );
	void step(double dt);
};

#endif // NBODY_SOLVER_ADAMS_H
