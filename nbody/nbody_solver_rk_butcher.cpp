#include "nbody_solver_rk_butcher.h"
#include <QDebug>

nbody_solver_rk_butcher::nbody_solver_rk_butcher( nbody_butcher_table* t ) :
	nbody_solver()
{
	m_k = NULL;
	m_tmpy = NULL;
	m_tmpk = NULL;
	m_y_stack = NULL;
	m_coeff = NULL;
	m_bt = t;
	m_step_subdivisions = 2;
	m_error_threshold = 1e-4;
}

nbody_solver_rk_butcher::~nbody_solver_rk_butcher()
{
	delete m_bt;
	engine()->free( m_k );;
	engine()->free( m_tmpy );
	engine()->free( m_tmpk );
	engine()->free( m_y_stack );
	engine()->free( m_coeff );
}

void nbody_solver_rk_butcher::step( double dt )
{
	nbody_engine::memory*	y = engine()->y();
	nbcoord_t				t = engine()->get_time();

	sub_step( 1, t, dt, y, 0, 0 );

	engine()->advise_time( dt );
}

void nbody_solver_rk_butcher::sub_step( size_t substeps_count, nbcoord_t t, nbcoord_t dt, nbody_engine::memory* y, size_t yoff, size_t recursion_level )
{
	const size_t		STEPS = m_bt->get_steps();
	const nbcoord_t**	a = m_bt->get_a();
	const nbcoord_t*	b1 = m_bt->get_b1();
	const nbcoord_t*	b2 = m_bt->get_b2();
	const nbcoord_t*	c = m_bt->get_c();
	size_t				ps = engine()->problem_size();
	size_t				coeff_count = STEPS+1;
	bool				need_first_approach_k = false;

	std::vector<nbcoord_t>	coeff;
	coeff.resize(coeff_count);

	if( m_k == NULL )
	{
		need_first_approach_k  = true;
		m_k = engine()->malloc( sizeof(nbcoord_t)*STEPS*ps );
		m_tmpy = engine()->malloc( sizeof(nbcoord_t)*ps );
		m_tmpk = engine()->malloc( sizeof(nbcoord_t)*ps );
		m_y_stack = engine()->malloc( sizeof(nbcoord_t)*ps*MAX_RECURSION );
		m_coeff = engine()->malloc( sizeof(nbcoord_t)*coeff_count );
	}

	for( size_t sub_n = 0; sub_n != substeps_count; ++sub_n, t += dt )
	{
		if( m_bt->is_implicit() )
		{
			size_t	max_iter = 3;

			if( need_first_approach_k )
			{
				//Compute first approach for <k>
				engine()->fcompute( t, y, m_tmpk, yoff, 0 );
				for( size_t i = 0; i != STEPS; ++i )
				{
					engine()->fmadd( m_tmpy, y, m_tmpk, dt*c[i], 0, yoff, 0 );
					engine()->fcompute( t + c[i]*dt, m_tmpy, m_k, 0, i*ps );
				}
			}

			//<k> iterative refinement
			for( size_t iter = 0; iter != max_iter; ++iter )
			{
				for( size_t i = 0; i != STEPS; ++i )
				{
					for( size_t n = 0; n != STEPS; ++n )
					{
						coeff.at(n) = dt*a[i][n];
					}
					engine()->memcpy( m_coeff, coeff.data() );
					engine()->fmaddn( m_tmpy, y, m_k, m_coeff, ps, 0, yoff, 0, STEPS );
					engine()->fcompute( t + c[i]*dt, m_tmpy, m_k, 0, i*ps );
				}
			}
		}
		else//Explicit method
		{
			for( size_t i = 0; i < STEPS; ++i )
			{
				if( i == 0 )
				{
					engine()->fcompute( t + c[i]*dt, y, m_k, yoff, i*ps );
				}
				else
				{
					for( size_t n = 0; n != i - 1; ++n )
					{
						coeff.at(n) = dt*a[i][n];
					}
					engine()->memcpy( m_coeff, coeff.data() );
					engine()->fmaddn( m_tmpy, y, m_k, m_coeff, ps, 0, yoff, 0, i );
					engine()->fcompute( t + c[i]*dt, m_tmpy, m_k, 0, i*ps );
				}
			}
		}

		nbcoord_t	max_error = 0;

		if( m_bt->is_embedded() )
		{
			for( size_t n = 0; n != STEPS; ++n )
			{
				coeff.at(n) = (b2[n] - b1[n]);
			}
			engine()->memcpy( m_coeff, coeff.data() );
			//engine()->fmaddn( m_tmpy, NULL, m_k, m_coeff, ps, 0, 0, 0, STEPS );
			engine()->fmaxabs( m_tmpy, max_error );
		}

//		qDebug() << max_error;
		bool can_subdivide = ( m_bt->is_embedded() && recursion_level < MAX_RECURSION ) && dt > get_min_step();
		bool need_subdivide = max_error > m_error_threshold;

		if( can_subdivide && need_subdivide )
		{
			nbcoord_t	new_dt = dt/m_step_subdivisions;

//			qDebug() << QString( "-" ).repeated(recursion_level) << "sub_step #" << sub_n << "ERR" << max_error << "Down to dt" << new_dt;

			engine()->memcpy( m_y_stack, y, recursion_level*ps, yoff );
			sub_step( m_step_subdivisions, t, new_dt, m_y_stack, recursion_level*ps, recursion_level + 1 );
			engine()->memcpy( y, m_y_stack, yoff, recursion_level*ps );
		}
		else
		{
			for( size_t n = 0; n != STEPS; ++n )
			{
				coeff.at(n) = b2[n]*dt;
			}
			engine()->memcpy( m_coeff, coeff.data() );
			engine()->fmaddn( y, m_k, m_coeff, ps, yoff, 0, STEPS );
		}
	}//for( size_t sub_n = 0; sub_n != substeps_count; ++sub_n )
}
