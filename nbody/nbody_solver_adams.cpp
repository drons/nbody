#include "nbody_solver_adams.h"
#include "nbody_solver_euler.h"
#include "summation.h"
#include <QDebug>

nbody_solver_adams::nbody_solver_adams( size_t rank ) : nbody_solver()
{
	m_starter = new nbody_solver_euler();
	m_f = NULL;
	m_coeff = NULL;
	m_rank = rank;
}

nbody_solver_adams::~nbody_solver_adams()
{
	delete m_starter;
	engine()->free_buffer( m_f );
	engine()->free_buffer( m_coeff );
}

const char* nbody_solver_adams::type_name() const
{
	return "nbody_solver_adams";
}

void nbody_solver_adams::advise( double dt )
{
	const nbcoord_t		a1[1] = { 1.0 };
	const nbcoord_t		a2[2] = { 3.0/2.0, -1.0/2.0 };
	const nbcoord_t		a3[3] = { 23.0/12.0, -4.0/3.0, 5.0/12.0 };
	const nbcoord_t		a4[4] = { 55.0/24.0, -59.0/24.0, 37.0/24.0, -3.0/8.0 };
	const nbcoord_t		a5[5] = { 1901.0/720.0, -1387.0/360.0, 109.0/30.0, -637.0/360.0, 251.0/720.0 };
	const nbcoord_t*	ar[] = { NULL, a1, a2, a3, a4, a5 };
	const nbcoord_t*	a = ar[m_rank];

	nbody_engine::memory*	y = engine()->get_y();
	nbcoord_t				t = engine()->get_time();
	size_t					step = engine()->get_step();
	size_t					fnum = step % m_rank;
	size_t					ps = engine()->problem_size();

	if( m_f== NULL )
	{
		m_starter->set_engine( engine() );
		m_f = engine()->create_buffer( sizeof( nbcoord_t )*ps*m_rank );
		m_coeff = engine()->create_buffer( sizeof( nbcoord_t )*m_rank );
	}

	if( step > m_rank )
	{
		std::vector<nbcoord_t>	coeff( m_rank );

		engine()->fcompute( t, y, m_f, 0, fnum*ps );

		for( size_t n = 0; n < m_rank; ++n )
		{
			coeff[ (m_rank+fnum-n)%m_rank ] = a[n]*dt;
		}

		engine()->write_buffer( m_coeff, coeff.data() );
		engine()->fmaddn_inplace( y, m_f, m_coeff, ps, 0, 0, m_rank );

		engine()->advise_time( dt );
	}
	else
	{
		engine()->fcompute( t, y, m_f, 0, fnum*ps );
		engine()->fmadd( y, y, m_f, dt, 0, 0, fnum*ps );
		engine()->advise_time( dt );
	}
}
