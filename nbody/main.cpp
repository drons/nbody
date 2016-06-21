#include <QApplication>
#include <QDebug>
#include <omp.h>

#include "wgt_nbody_view.h"

#include "nbody_solver_adams.h"
#include "nbody_solver_euler.h"
#include "nbody_solver_rkdp.h"
#include "nbody_solver_rkf.h"
#include "nbody_solver_rkgl.h"
#include "nbody_solver_rklc.h"
#include "nbody_solver_runge_kutta.h"
#include "nbody_solver_stormer.h"
#include "nbody_solver_trapeze.h"

#include "nbody_fcompute_block.h"
#include "nbody_fcompute_opencl.h"
#include "nbody_fcompute_simple.h"
#include "nbody_fcompute_sparse.h"

int gui_run( int argc, char *argv[], nbody_solver* solver, nbcoord_t box_size )
{
	QApplication	app( argc, argv );
	wgt_nbody_view*	nbv = new wgt_nbody_view( solver, box_size );

	nbv->show();

	QObject::connect( nbv, SIGNAL( destroyed() ),
					  &app, SLOT( quit() ) );

	double	update_time = omp_get_wtime();

	for(;;)
	{
		nbv->step();
		if( omp_get_wtime() - update_time > 0.05 )
		{
			nbv->updateGL();
			QApplication::processEvents();
			update_time = omp_get_wtime();
		}
	}
	return 0;
}

int con_run( int argc, char *argv[], nbody_solver* solver )
{
	QCoreApplication	a( argc, argv );
	solver->run( 1000, 0, 1 );
	return 0;
}

int main( int argc, char *argv[] )
{
	nbody_data              data;
	nbcoord_t				box_size = 100;

	data.make_universe( box_size, box_size, box_size );

	nbody_fcompute_opencl   engine;
	nbody_solver_rkdp		solver( &data );

	solver.set_time_step( 1e-9, 1e-2 );

	solver.set_engine( &engine );

	//return con_run( argc, argv, &solver );
	return gui_run( argc, argv, &solver, box_size );
}
