#include "nbody_solver_euler.h"
#include <QDebug>

nbody_solver_euler::nbody_solver_euler() : nbody_solver()
{
	m_dy = NULL;
}

nbody_solver_euler::~nbody_solver_euler()
{
	engine()->free_buffer(m_dy);
}

const char* nbody_solver_euler::type_name() const
{
	return "nbody_solver_euler";
}

void nbody_solver_euler::advise(nbcoord_t dt)
{
	nbody_engine::memory* y = engine()->get_y();

	if(m_dy == NULL)
	{
		m_dy = engine()->create_buffer(sizeof(nbcoord_t) * engine()->problem_size());
	}

	engine()->fcompute(engine()->get_time(), y, m_dy, 0, 0);
	engine()->fmadd_inplace(y, m_dy, dt);   //y += dy*dt

	engine()->advise_time(dt);
}
