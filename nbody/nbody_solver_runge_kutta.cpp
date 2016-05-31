#include "nbody_solver_runge_kutta.h"

nbody_solver_runge_kutta::nbody_solver_runge_kutta( nbody_data* data ) : nbody_solver( data )
{
}

void nbody_solver_runge_kutta::step( double dt )
{
	nbvertex_t*	vertites = data()->get_vertites();
	nbvertex_t*	velosites = data()->get_velosites();
	size_t		count = data()->get_count();

	if( k1.size() != count )
	{
		k1.resize( count );
	}
	if( k2.size() != count )
	{
		k2.resize( count );
	}
	if( k3.size() != count )
	{
		k3.resize( count );
	}
	if( k4.size() != count )
	{
		k4.resize( count );
	}

	if( q1.size() != count )
	{
		q1.resize( count );
	}
	if( q2.size() != count )
	{
		q2.resize( count );
	}
	if( q3.size() != count )
	{
		q3.resize( count );
	}
	if( q4.size() != count )
	{
		q4.resize( count );
	}

	if( tmp.size() != count )
	{
		tmp.resize( count );
	}

	step_v( vertites, k1.data() );
	#pragma omp parallel for
	for( size_t n = 0; n < count; ++n )
	{
		k1[n] *= dt;
		q1[n] = velosites[n]*dt;
		tmp[n] = vertites[n] + q1[n]*0.5;
	}

	step_v( tmp.data(), k2.data() );
	#pragma omp parallel for
	for( size_t n = 0; n < count; ++n )
	{
		k2[n] *= dt;
		q2[n] = (velosites[n] + k1[n]*0.5)*dt;
		tmp[n] = vertites[n] + q2[n]*0.5;
	}

	step_v( tmp.data(), k3.data() );
	#pragma omp parallel for
	for( size_t n = 0; n < count; ++n )
	{
		k3[n] *= dt;
		q3[n] = (velosites[n] + k2[n]*0.5)*dt;
		tmp[n] = vertites[n] + q3[n];
	}

	step_v( tmp.data(), k4.data() );
	#pragma omp parallel for
	for( size_t n = 0; n < count; ++n )
	{
		k4[n] *= dt;
		q4[n] = (velosites[n] + k3[n])*dt;

		velosites[n] += ( k1[n] + k2[n]*2.0 + k3[n]*2.0 + k4[n] )*(1.0/6.0);
		vertites[n] += ( q1[n] + q2[n]*2.0 + q3[n]*2.0 + q4[n] )*(1.0/6.0);
	}
	data()->advise_time( dt );
}
