#include "nbody_solver_midpoint.h"

nbody_solver_midpoint::nbody_solver_midpoint():
	m_k1(nullptr),
	m_k2(nullptr),
	m_tmp(nullptr)
{
}

nbody_solver_midpoint::~nbody_solver_midpoint()
{
	engine()->free_buffer(m_k1);
	engine()->free_buffer(m_k2);
	engine()->free_buffer(m_tmp);
}

const char* nbody_solver_midpoint::type_name() const
{
	return "nbody_solver_midpoint";
}

void nbody_solver_midpoint::advise(nbcoord_t dt)
{
	if(m_k1 == NULL)
	{
		m_k1 = engine()->create_buffer(sizeof(nbcoord_t) * engine()->problem_size());
		m_k2 = engine()->create_buffer(sizeof(nbcoord_t) * engine()->problem_size());
		m_tmp = engine()->create_buffer(sizeof(nbcoord_t) * engine()->problem_size());
	}
	nbody_engine::memory*	y = engine()->get_y();
	nbcoord_t				t = engine()->get_time();

	engine()->fcompute(t, y, m_k1);
	engine()->fmadd(m_tmp, y, m_k1, dt / 2_f);

	engine()->fcompute(t + dt / 2_f, m_tmp, m_k2);
	engine()->fmadd_inplace(y, m_k2, dt);

	engine()->advise_time(dt);
}
