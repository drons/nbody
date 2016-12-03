#include <QApplication>
#include <QDebug>
#include <omp.h>

#include "wgt_nbody_view.h"

#include "nbody_solvers.h"
#include "nbody_engines.h"

int gui_run( int argc, char *argv[], nbody_solver* solver, nbody_data* data, nbcoord_t box_size )
{
	QApplication	app( argc, argv );
	wgt_nbody_view*	nbv = new wgt_nbody_view( solver, data, box_size );

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

int con_run( int argc, char *argv[], nbody_solver* solver, nbody_data* data )
{
	QCoreApplication	a( argc, argv );
	solver->run( data, NULL, 3, 0, 1 );
	return 0;
}

int main( int argc, char *argv[] )
{
	nbody_data              data;
	nbcoord_t				box_size = 100;
	size_t					stars_count = 64;

	data.make_universe( stars_count, box_size, box_size, box_size );

	nbody_engine_simple		engine;
	nbody_solver_rk4		solver;

	engine.init( &data );
	solver.set_time_step( 1e-9, 1e-2 );
	solver.set_engine( &engine );

	//return con_run( argc, argv, &solver, &data );
	return gui_run( argc, argv, &solver, &data, box_size );
}
