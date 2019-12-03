#ifndef NBTYPE_H
#define NBTYPE_H

#ifndef NB_COORD_PRECISION
#define NB_COORD_PRECISION 2
#endif

#if NB_COORD_PRECISION == 1
typedef float nbcoord_t;
#elif NB_COORD_PRECISION == 2
typedef double nbcoord_t;
#elif NB_COORD_PRECISION == 4
#include "nbtype_quad.h"
typedef __float128 nbcoord_t;
#endif

#include "vertex.h"

typedef vertex3<nbcoord_t>		nbvertex_t;
typedef vertex4<float>			nbcolor_t;
typedef vertex3<float>			nb3f_t;
typedef vertex3<double>			nb3d_t;
typedef vertex4<float>			nb4f_t;
typedef vertex4<double>			nb4d_t;

constexpr nbcoord_t operator"" _f(unsigned long long int x)
{
	return static_cast<nbcoord_t>(x);
}

constexpr nbcoord_t operator"" _f(long double x)
{
	return static_cast<nbcoord_t>(x);
}

#define NBODY_DATA_BLOCK_SIZE	64
namespace nbody {
constexpr nbcoord_t MinDistance = 1e-8_f;	//!< Minimum distance between two bodyes
constexpr nbcoord_t GravityConstSI = 6.67259e-11_f;	// m^3/(kg s^2);
constexpr nbcoord_t Au = 149597870700_f;	//!< astronomical unit (m)
constexpr nbcoord_t Day = 86400_f;	//!< Julian day (s)
constexpr nbcoord_t GravityConstAuDayKg = (GravityConstSI* Day* Day) / (Au* Au* Au); // au^3/(kg day^2);
constexpr nbcoord_t MassFactorSI = GravityConstSI;
constexpr nbcoord_t MassFactorAuDayKg = GravityConstAuDayKg;
}

//! Types of Unit Systems
enum e_units_type
{
	//! Default units with identity gravity constant (G=1)
	eut_G1,
	//! Standart units (m-s-kg)
	eut_SI,
	//! Solar system units (au-day-kg)
	eut_au_day_kg
};

#endif // NBTYPE_H
