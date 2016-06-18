#include "nbody_solver_rkf.h"
#include <QDebug>

nbody_solver_rkf::nbody_solver_rkf( nbody_data* data ) :
	nbody_solver_rk_butcher( data, new nbody_butcher_table_rkf )
{
}

nbody_solver_rkf::~nbody_solver_rkf()
{
}
