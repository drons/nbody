#include "nbody_solver.h"
#include "summation.h"
#include <QDebug>

nbody_solver::nbody_solver( nbody_data* data )
	: m_data( data ), m_engine( NULL )
{
}

nbody_solver::~nbody_solver()
{

}

nbody_data* nbody_solver::data() const
{
	return m_data;
}

void nbody_solver::set_engine( nbody_fcompute* engine )
{
	m_engine = engine;
}

void nbody_solver::step_v( const nbvertex_t* vertites, nbvertex_t* dv )
{
	m_engine->fcompute( m_data, vertites, dv );
}

