#include "nbody_solver.h"
#include "nbody_data_stream.h"
#include <QDebug>

nbody_solver::nbody_solver()
	: m_engine(NULL), m_min_step(0), m_max_step(0)
{
}

nbody_solver::~nbody_solver()
{

}

void nbody_solver::set_engine(nbody_engine* e)
{
	m_engine = e;
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

int nbody_solver::run(nbody_data* data, nbody_data_stream* stream, nbcoord_t max_time, nbcoord_t dump_dt,
					  nbcoord_t check_dt)
{
	nbcoord_t	dt = get_max_step();
	nbcoord_t   last_check = 0;
	nbcoord_t   last_dump = 0;

	if(stream != NULL)
	{
		if(0 != stream->write(m_engine))
		{
			qDebug() << "Can't stream->write";
			return -1;
		}
		last_dump = data->get_time();
	}

	while(data->get_time() < max_time)
	{
		advise(dt);

		nbcoord_t   t = data->get_time();

		if(check_dt > 0 && t >= last_check + check_dt - dt * 0.1)
		{
			data->print_statistics(m_engine);
			last_check = t;
		}

		if(stream != NULL && dump_dt > 0 && t >= last_dump + dump_dt - dt * 0.1)
		{
			if(0 != stream->write(m_engine))
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
	qDebug() << "\tmin_step" << m_min_step;
	qDebug() << "\tmax_step" << m_max_step;
}

