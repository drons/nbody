#include <QApplication>
#include <omp.h>

#include "wgt_nbody_view.h"

int main(int argc, char *argv[] )
{
	QApplication	app( argc, argv );
	wgt_nbody_view*	nbv = new wgt_nbody_view();

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
