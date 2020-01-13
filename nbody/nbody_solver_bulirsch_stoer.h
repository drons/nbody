#ifndef NBODY_SOLVER_BULIRSCH_STOER_H
#define NBODY_SOLVER_BULIRSCH_STOER_H

#include "nbody_solver.h"

class nbody_extrapolator;

enum e_bs_sub { ebssub_bulirsch_stoer, ebssub_deuflhard };

/*!
	Bulirsch-Stoer solver
*/
class NBODY_DLL nbody_solver_bulirsch_stoer : public nbody_solver
{
	nbody_solver*				m_internal;
	size_t						m_max_level;
	nbcoord_t					m_error_threshold;
	std::vector<size_t>			m_sub_steps_count;
	nbody_engine::memory*		m_y0;
	nbody_extrapolator*			m_extrapolator;
public:
	explicit nbody_solver_bulirsch_stoer(size_t max_level = 8);
	~nbody_solver_bulirsch_stoer();
	const char* type_name() const override;
	void advise(nbcoord_t dt) override;
private:
	void compute_substep(size_t level, nbcoord_t dt, nbcoord_t t0);
};

#endif // NBODY_SOLVER_BULIRSCH_STOER_H
