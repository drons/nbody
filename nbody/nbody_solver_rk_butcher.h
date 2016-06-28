#ifndef NBODY_SOLVER_RK_BUTCHER_H
#define NBODY_SOLVER_RK_BUTCHER_H

#include "nbody_solver.h"
#include "nbody_butcher_table.h"

class nbody_solver_rk_butcher : public nbody_solver
{
	static const size_t			MAX_RECURSION = 8;
	nbody_butcher_table*		m_bt;
	nbody_engine::memory*		m_k;
	nbody_engine::memory*		m_tmpy;
	nbody_engine::memory*		m_tmpk;
	nbody_engine::memory*		m_coeff;
	nbody_engine::memory*		m_y_stack;
	int							m_step_subdivisions;
	nbcoord_t					m_error_threshold;
public:
	nbody_solver_rk_butcher( nbody_butcher_table* );
	~nbody_solver_rk_butcher();
	virtual void step( double dt );
private:
	void sub_step( size_t substeps_count, nbcoord_t t, nbcoord_t dt, nbody_engine::memory* y, size_t yoff, size_t recursion_level );
};

#endif // NBODY_SOLVER_RK_BUTCHER_H
