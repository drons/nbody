#ifndef SUMMATION_PROXY_H
#define SUMMATION_PROXY_H

#include "nbody_data.h"

struct impulce_proxy
{
	const nbvertex_t*	vel;
	const nbcoord_t*	m;
	impulce_proxy( nbody_data* d ) :
		vel( d->get_velosites() ),
		m( d->get_mass() )
	{
	}
	nbvertex_t operator [] ( size_t n ) const
	{
		return vel[n]*m[n];
	}
};

struct mass_center_proxy
{
	const nbvertex_t*	pos;
	const nbcoord_t*	m;
	mass_center_proxy( nbody_data* d ) :
		pos( d->get_vertites() ),
		m( d->get_mass() )
	{
	}
	nbvertex_t operator [] ( size_t n ) const
	{
		return pos[n]*m[n];
	}
};

struct impulce_moment_proxy
{
	const nbvertex_t*	pos;
	const nbvertex_t*	vel;
	const nbcoord_t*	m;
	impulce_moment_proxy( nbody_data* d ) :
		pos( d->get_vertites() ),
		vel( d->get_velosites() ),
		m( d->get_mass() )
	{
	}
	nbvertex_t operator [] ( size_t n ) const
	{
		return pos[n]^( vel[n]*m[n] );
	}
};

struct kinetic_energy_proxy
{
	const nbvertex_t*	vel;
	const nbcoord_t*	m;
	kinetic_energy_proxy( nbody_data* d ) :
		vel( d->get_velosites() ),
		m( d->get_mass() )
	{
	}
	nbcoord_t operator [] ( size_t n ) const
	{
		return vel[n].norm()*m[n];
	}
};

struct potential_energy_proxy
{
	const nbvertex_t*	vertites;
	const nbody_data*	data;
	potential_energy_proxy( nbody_data* d ) :
		vertites( d->get_vertites() ),
		data( d )
	{
	}
	nbcoord_t operator [] ( size_t n ) const
	{
		size_t	n1 = n / data->get_count();
		size_t	n2 = n % data->get_count();
		if( n1 == n2 )
		{
			return 0;
		}
		return data->potential_energy( vertites, n1, n2 );
	}
};

#endif // SUMMATION_PROXY_H
