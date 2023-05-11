#include "nbody_solver.h"
#include "nbody_data_stream.h"
#include "nbody_step_visitor.h"
#include <QDebug>

nbody_solver::nbody_solver()
	: m_engine(NULL),
	  m_min_step(0),
	  m_max_step(0),
	  m_clamp_to_box(false)
{
}

nbody_solver::~nbody_solver()
{
}

void nbody_solver::set_engine(nbody_engine* e)
{
	m_engine = e;
	m_engine->set_ode_order(get_ode_order());
}

nbody_engine* nbody_solver::engine()
{
	return m_engine;
}

void nbody_solver::set_time_step(nbcoord_t min_step, nbcoord_t max_step)
{
	m_min_step = min_step;
	m_max_step = max_step;
}

nbcoord_t nbody_solver::get_min_step() const
{
	return m_min_step;
}

nbcoord_t nbody_solver::get_max_step() const
{
	return m_max_step;
}

void nbody_solver::set_clamp_to_box(bool clamp)
{
	m_clamp_to_box = clamp;
}

void nbody_solver::add_check_visitor(std::shared_ptr<nbody_step_visitor> v)
{
	m_check_visitors.push_back(v);
}

int nbody_solver::run(nbody_data* data, nbody_data_stream* stream, nbcoord_t max_time,
					  nbcoord_t dump_dt, nbcoord_t check_dt)
{
	nbcoord_t	dt = get_max_step();
	nbcoord_t   last_check = data->get_time();
	nbcoord_t   last_dump = last_check;

	if(stream != NULL && dump_dt > 0 && last_dump <= 0)
	{
		m_engine->get_data(data);
		if(0 != stream->write(data))
		{
			qDebug() << "Can't stream->write";
			return -1;
		}
		last_dump = data->get_time();
	}

	while(data->get_time() < max_time)
	{
		if(m_clamp_to_box)
		{
			m_engine->clamp(m_engine->get_y(), data->get_box_size());
		}
		advise(dt);

		nbcoord_t   t = data->get_time();

		if(check_dt > 0 && t >= last_check + check_dt - dt * 0.1)
		{
			data->print_statistics(m_engine);
			for(auto v : m_check_visitors)
			{
				v->visit(data);
			}
			last_check = t;
		}

		if(stream != NULL && dump_dt > 0 && t >= last_dump + dump_dt - dt * 0.1)
		{
			m_engine->get_data(data);
			if(0 != stream->write(data))
			{
				qDebug() << "Can't stream->write";
				return -1;
			}
			last_dump = t;
		}
	}
	return 0;
}

void nbody_solver::print_info() const
{
	qDebug() << "\tmin_step: " << m_min_step;
	qDebug() << "\tmax_step: " << m_max_step;
	qDebug() << "\tODE order:" << get_ode_order();
	qDebug() << "\tClamp:    " << m_clamp_to_box;
}

e_ode_order nbody_solver::get_ode_order() const
{
	return eode_first_order;
}

void nbody_solver::reset()
{
}

