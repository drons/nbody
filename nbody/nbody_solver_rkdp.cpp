#include "nbody_solver_rkdp.h"
#include <QDebug>

nbody_solver_rkdp::nbody_solver_rkdp( nbody_data* data ) :
	nbody_solver_rk_butcher( data, new nbody_butcher_table_rkdp )
{
}

nbody_solver_rkdp::~nbody_solver_rkdp()
{
}
