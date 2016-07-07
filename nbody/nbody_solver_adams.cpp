#include "nbody_solver_adams.h"
#include "nbody_solver_euler.h"
#include "summation.h"
#include <QDebug>

nbody_solver_adams::nbody_solver_adams() : nbody_solver()
{
	m_starter = new nbody_solver_euler();
	m_f = NULL;
	m_coeff = NULL;
}

nbody_solver_adams::~nbody_solver_adams()
{
	delete m_starter;
	engine()->free( m_f );
	engine()->free( m_coeff );
}

void nbody_solver_adams::step( double dt )
{
	const size_t		rank = 5;
	const nbcoord_t		a1[1] = { 1.0 };
	const nbcoord_t		a2[2] = { 3.0/2.0, -1.0/2.0 };
	const nbcoord_t		a3[3] = { 23.0/12.0, -4.0/3.0, 5.0/12.0 };
	const nbcoord_t		a4[4] = { 55.0/24.0, -59.0/24.0, 37.0/24.0, -3.0/8.0 };
	const nbcoord_t		a5[5] = { 1901.0/720.0, -1387.0/360.0, 109.0/30.0, -637.0/360.0, 251.0/720.0 };
	const nbcoord_t*	ar[] = { NULL, a1, a2, a3, a4, a5 };
	const nbcoord_t*	a = ar[rank];

	nbody_engine::memory*	y = engine()->y();
	nbcoord_t				t = engine()->get_time();
	size_t					step = engine()->get_step();
	size_t					fnum = step % rank;
	size_t					ps = engine()->problem_size();

	if( m_f== NULL )
	{
		m_starter->set_engine( engine() );
		m_f = engine()->malloc( sizeof( nbcoord_t )*ps*rank );
		m_coeff = engine()->malloc( sizeof( nbcoord_t )*rank );
	}

	if( step > rank )
	{
		std::vector<nbcoord_t>	coeff( rank );

		engine()->fcompute( t, y, m_f, 0, fnum*ps );

		for( size_t n = 0; n < rank; ++n )
		{
			coeff[ (rank+fnum-n)%rank ] = a[n]*dt;
		}

		engine()->memcpy( m_coeff, coeff.data() );
		engine()->fmaddn( y, m_f, m_coeff, ps, 0, 0, rank );

		engine()->advise_time( dt );
	}
	else
	{
		engine()->fcompute( t, y, m_f, 0, fnum*ps );
		engine()->fmadd( y, y, m_f, dt, 0, 0, fnum*ps );
		engine()->advise_time( dt );
	}
}
