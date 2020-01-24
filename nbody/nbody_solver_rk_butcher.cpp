#include "nbody_solver_rk_butcher.h"
#include <QDebug>

nbody_solver_rk_butcher::nbody_solver_rk_butcher(nbody_butcher_table* t) :
	nbody_solver(),
	m_bt(t),
	m_tmpy(nullptr),
	m_tmpk(nullptr),
	m_corr_data(nullptr),
	m_max_recursion(8),
	m_substep_subdivisions(8),
	m_error_threshold(1e-4),
	m_refine_steps_count(1),
	m_correction(false)
{
}

nbody_solver_rk_butcher::~nbody_solver_rk_butcher()
{
	delete m_bt;
	engine()->free_buffers(m_k);;
	engine()->free_buffer(m_tmpy);
	engine()->free_buffer(m_tmpk);
	engine()->free_buffer(m_corr_data);
	engine()->free_buffers(m_y_stack);
}

const char* nbody_solver_rk_butcher::type_name() const
{
	return "nbody_solver_rk_butcher";
}

void nbody_solver_rk_butcher::set_max_recursion(size_t v)
{
	m_max_recursion = v;
}

void nbody_solver_rk_butcher::set_substep_subdivisions(size_t v)
{
	m_substep_subdivisions = v;
}

void nbody_solver_rk_butcher::set_error_threshold(nbcoord_t v)
{
	m_error_threshold = v;
}

void nbody_solver_rk_butcher::set_refine_steps_count(size_t v)
{
	m_refine_steps_count = v;
}

void nbody_solver_rk_butcher::set_correction(bool corr)
{
	m_correction = corr;
}

void nbody_solver_rk_butcher::advise(nbcoord_t dt)
{
	nbody_engine::memory*	y = engine()->get_y();
	nbcoord_t				t = engine()->get_time();

	sub_step(1, t, dt, y, 0);

	engine()->advise_time(dt);
}

void nbody_solver_rk_butcher::print_info() const
{
	nbody_solver::print_info();
	qDebug() << "\tmax_recursion" << m_max_recursion;
	qDebug() << "\tsubstep_subdivisions" << m_substep_subdivisions;
	qDebug() << "\terror_threshold" << m_error_threshold;
	qDebug() << "\trefine_steps_count" << m_refine_steps_count;
	qDebug() << "\tcorrection" << m_correction;
}

void nbody_solver_rk_butcher::reset()
{
	if(m_corr_data != nullptr)
	{
		engine()->fill_buffer(m_corr_data, 0);
	}
}

const nbody_butcher_table* nbody_solver_rk_butcher::table() const
{
	return m_bt;
}

void nbody_solver_rk_butcher::sub_step_implicit(size_t steps, const nbcoord_t** a,
												nbcoord_t* coeff,
												const nbody_engine::memory* y,
												bool need_first_approach_k,
												const nbcoord_t* c, nbcoord_t t,
												nbcoord_t dt)
{
	if(need_first_approach_k)
	{
		//Compute first approach for <k>
		engine()->fcompute(t, y, m_tmpk);
		for(size_t i = 0; i != steps; ++i)
		{
			engine()->fmadd(m_tmpy, y, m_tmpk, dt * c[i]);
			engine()->fcompute(t + c[i]*dt, m_tmpy, m_k[i]);
		}
	}

	//<k> iterative refinement
	for(size_t iter = 0; iter != m_refine_steps_count; ++iter)
	{
		for(size_t i = 0; i != steps; ++i)
		{
			for(size_t n = 0; n != steps; ++n)
			{
				coeff[n] = dt * a[i][n];
			}
			engine()->fmaddn(m_tmpy, y, m_k, coeff, m_k.size());
			engine()->fcompute(t + c[i]*dt, m_tmpy, m_k[i]);
		}
	}
}

void nbody_solver_rk_butcher::sub_step_explicit(size_t steps, const nbcoord_t** a,
												nbcoord_t* coeff,
												const nbody_engine::memory* y,
												const nbcoord_t* c, nbcoord_t t,
												nbcoord_t dt)
{
	for(size_t i = 0; i < steps; ++i)
	{
		if(i == 0)
		{
			engine()->fcompute(t + c[i]*dt, y, m_k[i]);
		}
		else
		{
			for(size_t n = 0; n != i; ++n)
			{
				coeff[n] = dt * a[i][n];
			}
			engine()->fmaddn(m_tmpy, y, m_k, coeff, i);
			engine()->fcompute(t + c[i]*dt, m_tmpy, m_k[i]);
		}
	}
}

void nbody_solver_rk_butcher::sub_step(size_t substeps_count, nbcoord_t t, nbcoord_t dt,
									   nbody_engine::memory* y, size_t recursion_level)
{
	const size_t		steps = m_bt->get_steps();
	const nbcoord_t**	a = m_bt->get_a();
	const nbcoord_t*	b1 = m_bt->get_b1();
	const nbcoord_t*	b2 = m_bt->get_b2();
	const nbcoord_t*	c = m_bt->get_c();
	size_t				ps = engine()->problem_size();
	size_t				coeff_count = steps + 1;
	bool				need_first_approach_k = false;

	std::vector<nbcoord_t>	coeff;
	coeff.resize(coeff_count);

	if(m_k.empty())
	{
		need_first_approach_k  = true;
		m_k = engine()->create_buffers(sizeof(nbcoord_t) * ps, steps);
		m_tmpy = engine()->create_buffer(sizeof(nbcoord_t) * ps);
		m_tmpk = engine()->create_buffer(sizeof(nbcoord_t) * ps);
		m_y_stack = engine()->create_buffers(sizeof(nbcoord_t) * ps, m_max_recursion);
		if(m_correction)
		{
			m_corr_data = engine()->create_buffer(sizeof(nbcoord_t) * ps);
			engine()->fill_buffer(m_corr_data, 0);
		}
	}

	for(size_t sub_n = 0; sub_n != substeps_count; ++sub_n, t += dt)
	{
		if(m_bt->is_implicit())
		{
			sub_step_implicit(steps, a, coeff.data(), y, need_first_approach_k, c, t, dt);
		}
		else
		{
			sub_step_explicit(steps, a, coeff.data(), y, c, t, dt);
		}

		nbcoord_t	max_error = 0;

		if(m_bt->is_embedded())
		{
			for(size_t n = 0; n != steps; ++n)
			{
				coeff[n] = (b2[n] - b1[n]);
			}
			engine()->fmaddn(m_tmpy, NULL, m_k, coeff.data(), steps);
			engine()->fmaxabs(m_tmpy, max_error);
		}

//		qDebug() << max_error;
		bool can_subdivide = (m_bt->is_embedded() && recursion_level < m_max_recursion) && dt > get_min_step();
		bool need_subdivide = max_error > m_error_threshold;

		if(can_subdivide && need_subdivide)
		{
			nbcoord_t	new_dt = dt / static_cast<nbcoord_t>(m_substep_subdivisions);

//			qDebug() << QString( "-" ).repeated(recursion_level) << "sub_step #" << sub_n << "ERR" << max_error << "Down to dt" << new_dt;

			nbody_engine::memory*	curr_y = m_y_stack[recursion_level];
			engine()->copy_buffer(curr_y, y);
			sub_step(m_substep_subdivisions, t, new_dt, curr_y, recursion_level + 1);
			engine()->copy_buffer(y, curr_y);
		}
		else
		{
			for(size_t n = 0; n != steps; ++n)
			{
				coeff[n] = b2[n] * dt;
			}
			if(m_correction)
			{
				engine()->fmaddn_corr(y, m_corr_data, m_k, coeff.data());
			}
			else
			{
				engine()->fmaddn_inplace(y, m_k, coeff.data());
			}
		}
	}//for( size_t sub_n = 0; sub_n != substeps_count; ++sub_n )
}
