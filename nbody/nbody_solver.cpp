#include "nbody_solver.h"
#include <QDebug>

nbody_solver::nbody_solver()
	: m_engine( NULL ), m_min_step(0), m_max_step(0)
{
}

nbody_solver::~nbody_solver()
{

}

void nbody_solver::set_engine( nbody_engine* engine )
{
	m_engine = engine;
}

nbody_engine* nbody_solver::engine()
{
	return m_engine;
}

void nbody_solver::step_v( const nbvertex_t* vertites, nbvertex_t* dv )
{
//	m_engine->fcompute( m_data, vertites, dv );
}

void nbody_solver::set_time_step( nbcoord_t min_step, nbcoord_t max_step )
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

int nbody_solver::run( nbody_data* data, nbcoord_t max_time, nbcoord_t dump_dt, nbcoord_t check_dt )
{
	nbcoord_t	dt = get_max_step();
	nbcoord_t   last_check = 0;
	nbcoord_t   last_dump = 0;
	while( data->get_time() < max_time )
	{
		nbcoord_t   t = data->get_time();
		if( check_dt > 0 && t >= last_check + check_dt - dt*0.1 )
		{
			data->print_statistics( m_engine );
			last_check = t;
		}
		if( dump_dt > 0 && t >= last_dump + dump_dt - dt*0.1 )
		{
			data->dump();
			last_dump = t;
		}

		step( dt );
	}
	return 0;
}

