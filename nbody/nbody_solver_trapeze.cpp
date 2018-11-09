#include "nbody_solver_trapeze.h"
#include "summation.h"

nbody_solver_trapeze::nbody_solver_trapeze() : nbody_solver()
{
	m_predictor = NULL;
	m_refine_steps_count = 1;
}

nbody_solver_trapeze::~nbody_solver_trapeze()
{
	engine()->free_buffers(m_f);
	engine()->free_buffer(m_predictor);
}

const char* nbody_solver_trapeze::type_name() const
{
	return "nbody_solver_trapeze";
}

void nbody_solver_trapeze::set_refine_steps_count(size_t v)
{
	m_refine_steps_count = v;
}

void nbody_solver_trapeze::advise(nbcoord_t dt)
{
	nbody_engine::memory*	y = engine()->get_y();
	nbcoord_t				t = engine()->get_time();
	size_t					ps = engine()->problem_size();

	if(m_f.empty())
	{
		m_f = engine()->create_buffers(sizeof(nbcoord_t) * ps, 2);
		m_predictor = engine()->create_buffer(sizeof(nbcoord_t) * ps);
	}

	engine()->fcompute(t, y, m_f[0]);
	engine()->fmadd(m_predictor, y, m_f[0], dt);

	for(size_t s = 0; s <= m_refine_steps_count; ++s)
	{
		const nbcoord_t	coeff[] = { dt / 2, dt / 2 };
		engine()->fcompute(t, m_predictor, m_f[1]);

		if(s == m_refine_steps_count)
		{
			engine()->fmaddn_inplace(y, m_f, coeff);
		}
		else
		{
			engine()->fmaddn(m_predictor, y, m_f, coeff, m_f.size());
		}
	}
	engine()->advise_time(dt);
}

void nbody_solver_trapeze::print_info() const
{
	nbody_solver::print_info();
	qDebug() << "\trefine_steps_count" << m_refine_steps_count;
}
