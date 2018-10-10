#ifndef NBODY_SOLVER_ADAMS_H
#define NBODY_SOLVER_ADAMS_H

#include "nbody_solver.h"

/*!
   \brief Adams–Bashforth method
 */
class nbody_solver_adams : public nbody_solver
{
	nbody_solver*			m_starter;
	nbody_engine::memory*	m_f;
	nbody_engine::memory*	m_coeff;
	size_t					m_rank;
public:
	explicit nbody_solver_adams(size_t rank = 5);
	~nbody_solver_adams();
	const char* type_name() const override;
	void advise(double dt) override;
};

#endif // NBODY_SOLVER_ADAMS_H
