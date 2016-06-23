#include "nbody_solver_euler.h"
#include <QDebug>

nbody_solver_euler::nbody_solver_euler() : nbody_solver( NULL )
{
	m_dy = NULL;
}

nbody_solver_euler::~nbody_solver_euler()
{
	engine()->free( m_dy );
}

void nbody_solver_euler::step( nbcoord_t dt )
{
	nbody_engine::memory* y = engine()->y();

	if( m_dy == NULL )
	{
		m_dy = engine()->malloc( sizeof(nbcoord_t)*engine()->problem_size() );
	}

	engine()->fcompute( 0, y, m_dy, 0 );
	engine()->fmadd( y, m_dy, dt ); //y += dy*dt

	engine()->advise_time( dt );
}
