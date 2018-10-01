#include <QApplication>
#include <QDebug>
#include <omp.h>

#include "wgt_nbody_player.h"

#include "nbody_solvers.h"
#include "nbody_engines.h"

int main(int argc, char* argv[])
{
	QApplication		app(argc, argv);
	nbcoord_t			box_size = 100;
	nbody_data			data;
	nbody_engine_simple	engine;
	nbody_solver_euler	solver;

	data.make_universe(1024, box_size, box_size, box_size);

	engine.init(&data);
	solver.set_engine(&engine);

	wgt_nbody_player*	nbv = new wgt_nbody_player(&solver, &data, box_size);

	nbv->show();

	return app.exec();
}
