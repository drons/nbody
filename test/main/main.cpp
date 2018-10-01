#include <QApplication>
#include <QDebug>
#include <omp.h>

#include "nbody_solvers.h"
#include "nbody_engines.h"
#include "nbody_data_stream.h"

int con_run(int argc, char* argv[], nbody_solver* solver, nbody_data* data)
{
	QCoreApplication	a(argc, argv);
	nbody_data_stream	stream;

	if(0 != stream.open("/tmp/nbody/main-stream", 1 << 30))
	{
		qDebug() << "Fail to open stream";
		return -1;
	}
	solver->run(data, &stream, 3, 1e-2, 1);
	return 0;
}

int main(int argc, char* argv[])
{
	nbody_data              data;
	nbcoord_t				box_size = 100;
	size_t					stars_count = 64;

	data.make_universe(stars_count, box_size, box_size, box_size);

	nbody_engine_simple		engine;
	nbody_solver_rk4		solver;

	engine.init(&data);
	solver.set_time_step(1e-9, 1e-2);
	solver.set_engine(&engine);

	return con_run(argc, argv, &solver, &data);
}
