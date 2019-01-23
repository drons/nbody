#ifndef NBODY_SOLVER_ADAMS_H
#define NBODY_SOLVER_ADAMS_H

#include "nbody_solver.h"

/*!
   \brief Adams–Bashforth method
 */
class NBODY_DLL nbody_solver_adams : public nbody_solver
{
	nbody_solver*				m_starter;
	nbody_engine::memory_array	m_f;
	size_t						m_rank;
public:
	explicit nbody_solver_adams(nbody_solver* starter, size_t rank = 5);
	~nbody_solver_adams();
	const char* type_name() const override;
	void advise(nbcoord_t dt) override;
	void print_info() const override;
};

#endif // NBODY_SOLVER_ADAMS_H
