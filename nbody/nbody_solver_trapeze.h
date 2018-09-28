#ifndef NBODY_SOLVER_TRAPEZE_H
#define NBODY_SOLVER_TRAPEZE_H

#include "nbody_solver.h"

class nbody_solver_trapeze : public nbody_solver
{
	nbody_engine::memory*	m_f01;
	nbody_engine::memory*	m_predictor;
	nbody_engine::memory*	m_coeff;
	size_t					m_refine_steps_count;
public:
	nbody_solver_trapeze();
	~nbody_solver_trapeze();
	const char* type_name() const override;
	void set_refine_steps_count( size_t );
	virtual void advise( nbcoord_t dt ) override;
};

#endif // NBODY_SOLVER_TRAPEZE_H
