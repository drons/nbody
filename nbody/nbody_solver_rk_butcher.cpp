#include "nbody_solver_rk_butcher.h"
#include <QDebug>

nbody_solver_rk_butcher::nbody_solver_rk_butcher(nbody_butcher_table* t) :
	nbody_solver()
{
	m_k = NULL;
	m_tmpy = NULL;
	m_tmpk = NULL;
	m_coeff = NULL;
	m_bt = t;
	m_max_recursion = 8;
	m_substep_subdivisions = 8;
	m_error_threshold = 1e-4;
	m_refine_steps_count = 1;
}

nbody_solver_rk_butcher::~nbody_solver_rk_butcher()
{
	delete m_bt;
	engine()->free_buffer(m_k);;
	engine()->free_buffer(m_tmpy);
	engine()->free_buffer(m_tmpk);
	for(auto b : m_y_stack)
	{
		engine()->free_buffer(b);
	}
	engine()->free_buffer(m_coeff);
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

void nbody_solver_rk_butcher::advise(nbcoord_t dt)
{
	nbody_engine::memory*	y = engine()->get_y();
	nbcoord_t				t = engine()->get_time();

	sub_step(1, t, dt, y, 0, 0);

	engine()->advise_time(dt);
}

void nbody_solver_rk_butcher::print_info() const
{
	nbody_solver::print_info();
	qDebug() << "\tmax_recursion" << m_max_recursion;
	qDebug() << "\tsubstep_subdivisions" << m_substep_subdivisions;
	qDebug() << "\terror_threshold" << m_error_threshold;
	qDebug() << "\trefine_steps_count" << m_refine_steps_count;
}

void nbody_solver_rk_butcher::sub_step(size_t substeps_count, nbcoord_t t, nbcoord_t dt, nbody_engine::memory* y,
									   size_t yoff, size_t recursion_level)
{
	const size_t		STEPS = m_bt->get_steps();
	const nbcoord_t**	a = m_bt->get_a();
	const nbcoord_t*	b1 = m_bt->get_b1();
	const nbcoord_t*	b2 = m_bt->get_b2();
	const nbcoord_t*	c = m_bt->get_c();
	size_t				ps = engine()->problem_size();
	size_t				coeff_count = STEPS + 1;
	bool				need_first_approach_k = false;

	std::vector<nbcoord_t>	coeff;
	coeff.resize(coeff_count);

	if(m_k == NULL)
	{
		need_first_approach_k  = true;
		m_k = engine()->create_buffer(sizeof(nbcoord_t) * STEPS * ps);
		m_tmpy = engine()->create_buffer(sizeof(nbcoord_t) * ps);
		m_tmpk = engine()->create_buffer(sizeof(nbcoord_t) * ps);
		m_y_stack.resize(m_max_recursion);
		for(size_t level = 0; level != m_max_recursion; ++level)
		{
			m_y_stack[level] = engine()->create_buffer(sizeof(nbcoord_t) * ps);
		}
		m_coeff = engine()->create_buffer(sizeof(nbcoord_t) * coeff_count);
	}

	for(size_t sub_n = 0; sub_n != substeps_count; ++sub_n, t += dt)
	{
		if(m_bt->is_implicit())
		{
			if(need_first_approach_k)
			{
				//Compute first approach for <k>
				engine()->fcompute(t, y, m_tmpk, yoff, 0);
				for(size_t i = 0; i != STEPS; ++i)
				{
					engine()->fmadd(m_tmpy, y, m_tmpk, dt * c[i], 0, yoff, 0);
					engine()->fcompute(t + c[i]*dt, m_tmpy, m_k, 0, i * ps);
				}
			}

			//<k> iterative refinement
			for(size_t iter = 0; iter != m_refine_steps_count; ++iter)
			{
				for(size_t i = 0; i != STEPS; ++i)
				{
					for(size_t n = 0; n != STEPS; ++n)
					{
						coeff.at(n) = dt * a[i][n];
					}
					engine()->write_buffer(m_coeff, coeff.data());
					engine()->fmaddn(m_tmpy, y, m_k, m_coeff, ps, 0, yoff, 0, STEPS);
					engine()->fcompute(t + c[i]*dt, m_tmpy, m_k, 0, i * ps);
				}
			}
		}
		else//Explicit method
		{
			for(size_t i = 0; i < STEPS; ++i)
			{
				if(i == 0)
				{
					engine()->fcompute(t + c[i]*dt, y, m_k, yoff, i * ps);
				}
				else
				{
					for(size_t n = 0; n != i; ++n)
					{
						coeff.at(n) = dt * a[i][n];
					}
					engine()->write_buffer(m_coeff, coeff.data());
					engine()->fmaddn(m_tmpy, y, m_k, m_coeff, ps, 0, yoff, 0, i);
					engine()->fcompute(t + c[i]*dt, m_tmpy, m_k, 0, i * ps);
				}
			}
		}

		nbcoord_t	max_error = 0;

		if(m_bt->is_embedded())
		{
			for(size_t n = 0; n != STEPS; ++n)
			{
				coeff.at(n) = (b2[n] - b1[n]);
			}
			engine()->write_buffer(m_coeff, coeff.data());
			engine()->fmaddn(m_tmpy, NULL, m_k, m_coeff, ps, 0, 0, 0, STEPS);
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
			engine()->copy_buffer(curr_y, y, 0, 0);
			sub_step(m_substep_subdivisions, t, new_dt, curr_y, 0, recursion_level + 1);
			engine()->copy_buffer(y, curr_y, 0, 0);
		}
		else
		{
			for(size_t n = 0; n != STEPS; ++n)
			{
				coeff.at(n) = b2[n] * dt;
			}
			engine()->write_buffer(m_coeff, coeff.data());
			engine()->fmaddn_inplace(y, m_k, m_coeff, ps, yoff, 0, STEPS);
		}
	}//for( size_t sub_n = 0; sub_n != substeps_count; ++sub_n )
}
