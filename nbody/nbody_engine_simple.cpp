#include "nbody_engine_simple.h"

nbody_engine_simple::nbody_engine_simple()
{
}

void nbody_engine_simple::fcompute( const nbody_data* data, const nbvertex_t* vertites, nbvertex_t* dv )
{
	advise_compute_count();

	size_t				count = data->get_count();
	const nbcoord_t*	mass = data->get_mass();

#pragma omp parallel for
	for( size_t body1 = 0; body1 < count; ++body1 )
	{
		const nbvertex_t&	v1( vertites[ body1 ] );
		nbvertex_t			total_force;
		for( size_t body2 = 0; body2 != count; ++body2 )
		{
			if( body1 == body2 )
				continue;
			const nbvertex_t&	v2( vertites[ body2 ] );
			const nbvertex_t	f( data->force( v1, v2, mass[body1], mass[body2] ) );
			total_force += f;
		}

		dv[body1] = total_force/mass[body1];
	}
}
