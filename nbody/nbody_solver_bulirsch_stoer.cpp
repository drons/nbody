#include "nbody_solver_bulirsch_stoer.h"
#include "nbody_solver_midpoint.h"
#include "nbody_extrapolator.h"
#include <QDebug>

namespace {
std::vector<size_t> init_steps(size_t n, e_bs_sub type)
{
	std::vector<size_t>	steps;
	if(n == 0)
	{
		return steps;
	}
	if(ebssub_bulirsch_stoer == type)
	{
		steps.push_back(2);
		if(n > 1)
		{
			steps.push_back(4);
		}
		if(n > 2)
		{
			steps.push_back(6);
		}
		for(size_t j = 3; j < n; ++j)
		{
			steps.push_back(2 * steps[j - 2]);
		}
	}
	else
	{
		for(size_t j = 0; j < n; ++j)
		{
			steps.push_back(2 * (j + 1));
		}
	}
	return steps;
}
}//namespace

nbody_solver_bulirsch_stoer::nbody_solver_bulirsch_stoer(size_t max_level) :
	nbody_solver(),
	m_internal(nullptr),
	m_max_level(max_level),
	m_error_threshold(1e-11),
	m_sub_steps_count(init_steps(m_max_level, ebssub_bulirsch_stoer)),
	m_y0(nullptr),
	m_extrapolator(nullptr)
{
}

nbody_solver_bulirsch_stoer::~nbody_solver_bulirsch_stoer()
{
	engine()->free_buffer(m_y0);
	delete m_extrapolator;
	delete m_internal;
}

const char* nbody_solver_bulirsch_stoer::type_name() const
{
	return "nbody_solver_bulirsch_stoer";
}

void nbody_solver_bulirsch_stoer::compute_substep(size_t level, nbcoord_t dt, nbcoord_t t0)
{
	nbody_engine::memory*	y = engine()->get_y();

	engine()->copy_buffer(y, m_y0);
	engine()->set_time(t0);

	// Compute ODE solution with step 'dt / substeps_count'
	size_t		substeps_count = m_sub_steps_count[level];
	nbcoord_t	substep_dt = dt / substeps_count;
	for(size_t substep_n = 0; substep_n != substeps_count; ++substep_n)
	{
		m_internal->advise(substep_dt);
	}

	m_extrapolator->update_table(level, y);
}

void nbody_solver_bulirsch_stoer::advise(nbcoord_t dt)
{
	if(m_y0 == nullptr)
	{
		size_t size = sizeof(nbcoord_t) * engine()->problem_size();
		m_y0 = engine()->create_buffer(size);
		m_extrapolator = nbody_create_extrapolator("neville", engine(), 2, m_sub_steps_count);
		m_internal = new nbody_solver_midpoint();
		m_internal->set_engine(engine());
	}

	nbody_engine::memory*	y = engine()->get_y();
	nbcoord_t				t0(engine()->get_time());

	engine()->copy_buffer(m_y0, y);

	for(size_t level = 0; level != m_max_level; ++level)
	{
//		QDebug	g(qDebug());
//		g << level << engine()->get_time() << t0;
		compute_substep(level, dt, t0);

		if(level < 2)
		{
			continue;
		}
		nbcoord_t	maxdiff = m_extrapolator->estimate_error(level);
//		g << engine()->get_time() << t0 << dt << maxdiff;
		if(maxdiff < m_error_threshold || level == m_max_level - 1)
		{
			m_extrapolator->extrapolate(level, y);
			break;
		}
	}

	engine()->set_time(t0);
	engine()->advise_time(dt);
}
