#include "nbody_data.h"
#include "nbody_engine.h"
#include <qnumeric.h>
#include <QDebug>

#include "summation.h"
#include "summation_proxy.h"

#include <omp.h>

namespace {
	nbcoord_t G = 1;
}

nbody_data::nbody_data()
{
	m_time = 0;
	m_step = 0;
	m_count = 0;
	m_total_kinetic_energy = 0;
	m_total_potential_energy = 0;
	m_last_total_kinetic_energy = 0;
	m_last_total_potential_energy = 0;
	m_timer_start = omp_get_wtime();
	m_timer_step = 0;
}

nbvertex_t nbody_data::force( const nbvertex_t& v1, const nbvertex_t& v2, nbcoord_t mass1, nbcoord_t mass2) const
{
	nbvertex_t	dr( v1 - v2 );
	nbcoord_t	r2( dr.norm() );
	if( r2 < NBODY_MIN_R )
	{
		r2 = NBODY_MIN_R;
	}
	return dr*( (-G*mass1*mass2)/( r2*sqrt( r2 ) ) );
}

nbcoord_t nbody_data::potential_energy( const nbvertex_t* vertites, size_t body1, size_t body2 ) const
{
	nbvertex_t	dr( vertites[body1] - vertites[body2] );
	nbcoord_t	r2( dr.norm() );
	if( r2 < NBODY_MIN_R )
		return 0;
	return -(G*m_mass[body1]*m_mass[body2])/sqrt( r2 );
}

void nbody_data::print_statistics( nbody_engine* engine )
{
	double		timer_end = omp_get_wtime();
	nbvertex_t	total_impulce( summation<nbvertex_t,impulce_proxy>( impulce_proxy( this ), m_count ) );
	nbvertex_t	total_impulce_moment( summation<nbvertex_t,impulce_moment_proxy>( impulce_moment_proxy( this ), m_count ) );
	nbvertex_t	mass_center( summation<nbvertex_t, mass_center_proxy>( mass_center_proxy( this ), m_count ) );
	nbcoord_t	total_mass( summation<nbcoord_t, const nbcoord_t*>( m_mass.data(), m_count ) );
	nbcoord_t	total_kinetic_energy( summation<nbcoord_t,kinetic_energy_proxy>( kinetic_energy_proxy( this ), m_count ) );
	nbcoord_t	total_potential_energy( summation<nbcoord_t,potential_energy_proxy>( potential_energy_proxy( this ), m_count*m_count ));

	total_kinetic_energy /= 2;
	total_potential_energy /= 2;
	mass_center /= total_mass;

	static bool first_run = true;
	if( first_run )
	{
		m_total_impulce = total_impulce;
		m_total_impulce_moment = total_impulce_moment;
		m_mass_center = mass_center;
		m_total_kinetic_energy = total_kinetic_energy;
		m_total_potential_energy = total_potential_energy;
		first_run = false;
	}
	m_last_total_impulce = total_impulce;
	m_last_total_impulce_moment = total_impulce_moment;
	m_last_mass_center = mass_center;
	m_last_total_kinetic_energy = total_kinetic_energy;
	m_last_total_potential_energy = total_potential_energy;

	nbcoord_t	total_energy = total_potential_energy + total_kinetic_energy;
	total_impulce -= m_total_impulce;
	total_impulce_moment -= m_total_impulce_moment;
	mass_center -= m_mass_center;
	total_energy -=	(m_total_potential_energy + m_total_kinetic_energy);

	qDebug()<< "#" << m_step
			<< "t" << m_time
			<< "CC" << engine->get_compute_count()
			<< "dP" << impulce_err()
			<< "dL" << impulce_moment_err()
			<< "Vcm" << (mass_center/m_time).length()
			<< "dE" << energy_err()
			<< "St" << ( timer_end - m_timer_start )/( m_step - m_timer_step )
			<< "Wt" << ( omp_get_wtime() - m_timer_start )/( m_step - m_timer_step );

	m_timer_start = omp_get_wtime();
	m_timer_step = m_step;
}

void nbody_data::dump_body(size_t n)
{
	qDebug()<< "#" << n
			<< "M" << m_mass[n]
			<< "R" << m_vertites[n].x << m_vertites[n].y << m_vertites[n].z
			<< "V" << m_velosites[n].x << m_velosites[n].y << m_velosites[n].z;
}

void nbody_data::add_body(const nbvertex_t& r, const nbvertex_t& v, const nbcoord_t& m, const nbcoord_t& a, const nbcolor_t& color )
{
	m_vertites.push_back( r );
	m_velosites.push_back( v );
	m_mass.push_back( m );
	m_color.push_back( color );
	m_a.push_back( a );
	++m_count;
}

void nbody_data::advise_time( nbcoord_t dt )
{
	m_time += dt;
	++m_step;
}

nbcoord_t nbody_data::get_time() const
{
	return m_time;
}

size_t nbody_data::get_step() const
{
	return m_step;
}

nbvertex_t* nbody_data::get_vertites()
{
	return m_vertites.data();
}

const nbvertex_t* nbody_data::get_vertites() const
{
	return m_vertites.data();
}

nbvertex_t* nbody_data::get_velosites()
{
	return m_velosites.data();
}

const nbvertex_t* nbody_data::get_velosites() const
{
	return m_velosites.data();
}

const nbcoord_t* nbody_data::get_mass() const
{
	return m_mass.data();
}

const nbcolor_t* nbody_data::get_color() const
{
	return m_color.data();
}

size_t nbody_data::get_count() const
{
	return m_count;
}

nbcoord_t static randcoord( nbcoord_t min, nbcoord_t max )
{
	return min + (max - min)*((nbcoord_t)rand())/((nbcoord_t)RAND_MAX);
}

void nbody_data::add_galaxy( nbvertex_t center, nbvertex_t velosity, nbcoord_t radius, nbcoord_t total_mass, size_t count, nbcolor_t& color )
{
	count = (count/NBODY_DATA_BLOCK_SIZE+1)*NBODY_DATA_BLOCK_SIZE - 1;

	nbvertex_t	up( 0,0,1 );
	nbvertex_t	right( 1,0,0 );
	nbcoord_t	black_hole_mass_ratio = 0.999;
	nbcoord_t	black_hole_mass = total_mass*black_hole_mass_ratio;
	nbcoord_t	star_mass = ( total_mass - black_hole_mass )/((nbcoord_t)count);
	nbcoord_t	all_stars_mass = count*star_mass;

	add_body( center, velosity, black_hole_mass, 1, nbcolor_t( 0, 1, 0, 1 ) );

	for( size_t n = 0; n != count; ++n )
	{
		nbvertex_t	r(	randcoord( -radius, radius ),
						randcoord( -radius, radius ),
						randcoord( -radius, radius ) );
		nbcoord_t		rlen( r.length() );
		if( rlen > radius )
		{
			--n;
			continue;
		}
		r.z *= 0.3;
		r += center;
		nbvertex_t	v( (r - center)^up );
		if( v.length() < 1e-7 )
		{
			v = (r - center)^right;
		}
		v.normalize();
		nbcoord_t effective_mass = pow( rlen/radius, 3.0 )*all_stars_mass + black_hole_mass;
		v *=  sqrt( force( r, center, star_mass, effective_mass ).length()*( r - center ).length()/star_mass );
		add_body( r, v + velosity, star_mass, 1, color );
	}
}

void nbody_data::make_universe( nbcoord_t sx, nbcoord_t sy, nbcoord_t sz )
{
	nbcoord_t	radius = sx*0.5;
	nbcoord_t	galaxy_mass = 1000;
	size_t		star_count = 64;
	nbvertex_t	center( sx*0.5, sy*0.5, sz*0.5 );
	nbvertex_t	base( radius, 0, 0 );
	nbvertex_t	velosity( 0, sqrt(force( nbvertex_t(), base, galaxy_mass, galaxy_mass ).length()*(base).length()/(2*galaxy_mass)), 0 );
	srand(1);

	float		intensity = 0.8f;
	nbcolor_t	color1( intensity*0.5f, intensity*0.5f, intensity, 1.0f );
	nbcolor_t	color2( intensity, intensity*0.5f, intensity*0.5f, 1.0f );

	add_galaxy( center - base, velosity/3, radius, galaxy_mass, star_count, color1 );
	add_galaxy( center + base, -velosity/3, radius, galaxy_mass, star_count, color2 );
	//add_galaxy( center, vertex_t(), radius, galaxy_mass, star_count );
}

nbvertex_t nbody_data::total_impulce() const
{
	return m_total_impulce;
}

nbvertex_t nbody_data::total_impulce_moment() const
{
	return m_total_impulce_moment;
}

nbvertex_t nbody_data::mass_center() const
{
	return m_mass_center;
}

nbcoord_t nbody_data::total_energy() const
{
	return m_total_kinetic_energy + m_total_potential_energy;
}

nbvertex_t nbody_data::last_total_impulce() const
{
	return m_last_total_impulce;
}

nbvertex_t nbody_data::last_total_impulce_moment() const
{
	return m_last_total_impulce_moment;
}

nbvertex_t nbody_data::last_mass_center() const
{
	return m_last_mass_center;
}

nbcoord_t nbody_data::last_total_energy() const
{
	return m_last_total_kinetic_energy + m_last_total_potential_energy;
}

nbcoord_t nbody_data::impulce_err() const
{
	return fabs( 100.0*( m_last_total_impulce - m_total_impulce ).length()/m_total_impulce.length() );
}

nbcoord_t nbody_data::impulce_moment_err() const
{
	return fabs( 100.0*( m_last_total_impulce_moment - m_total_impulce_moment ).length()/m_total_impulce_moment.length() );
}

nbcoord_t nbody_data::energy_err() const
{
	return fabs( 100.0*( last_total_energy() - total_energy() )/total_energy() );
}

void nbody_data::dump()
{
	qDebug() << "Dump function placeholder";
}

