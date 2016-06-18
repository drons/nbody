#ifndef NBODY_SOLVER_RK_BUTCHER_H
#define NBODY_SOLVER_RK_BUTCHER_H

#include "nbody_solver.h"
#include "nbody_butcher_table.h"

class nbody_solver_rk_butcher : public nbody_solver
{
	static const size_t			MAX_RECURSION = 8;
	nbody_butcher_table*		m_bt;
	std::vector< std::vector< nbvertex_t > >	m_k;
	std::vector< std::vector< nbvertex_t > >	m_q;
	std::vector< nbvertex_t* >	m_kptr;
	std::vector< nbvertex_t* >	m_qptr;

	std::vector< nbvertex_t >	tmpvrt;
	std::vector< nbvertex_t >	tmpvel;
	std::vector< nbvertex_t >	vertites_stack[MAX_RECURSION];
	std::vector< nbvertex_t >	velosites_stack[MAX_RECURSION];
	int							m_step_subdivisions;
	nbcoord_t					m_vrt_error_threshold;
	nbcoord_t					m_vel_error_threshold;
public:
	nbody_solver_rk_butcher( nbody_data* data, nbody_butcher_table* );
	~nbody_solver_rk_butcher();
	virtual void step( double dt );
private:
	void sub_step( size_t substeps_count, nbcoord_t dt, nbvertex_t* vertites, nbvertex_t* velosites, size_t recursion_level );
};

#endif // NBODY_SOLVER_RK_BUTCHER_H
