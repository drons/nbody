#include <QApplication>
#include <QDebug>
#include <omp.h>

#include "nbody_solvers.h"
#include "nbody_engines.h"
#include "nbody_data_stream.h"
#include "nbody_arg_parser.h"

int con_run(int argc, char* argv[], nbody_solver* solver, nbody_data* data, const QVariantMap& param)
{
	QString		output(param.value("output", "/tmp/nbody/main-stream").toString());
	size_t		max_part_size(static_cast<size_t>(param.value("max_part_size", 1 << 30).toULongLong()));
	nbcoord_t	max_time = param.value("max_time", 1).toDouble();
	nbcoord_t	dump_step = param.value("dump_step", 1e-2).toDouble();
	nbcoord_t	check_step = param.value("check_step", 1e-1).toDouble();
	QString		check_list(param.value("check_list", "PLV").toString());

	QCoreApplication	a(argc, argv);
	nbody_data_stream	stream;

	data->set_check_list(check_list);

	if(param.value("verbose", "0").toInt() != 0)
	{
		qDebug() << "General:";
		qDebug() << "\tStars count:" << data->get_count();
		qDebug() << "\tBox size:" << data->get_box_size();
		qDebug() << "\toutput:" << output;
		qDebug() << "\tmax_part_size:" << max_part_size;
		qDebug() << "\tmax_time:" << max_time;
		qDebug() << "\tdump_step:" << dump_step;
		qDebug() << "\tcheck_step:" << check_step;
		qDebug() << "\tcheck_list:" << check_list;
		qDebug() << "Solver:" << solver->type_name();
		solver->print_info();
		qDebug() << "Engine:" << solver->engine()->type_name();
		solver->engine()->print_info();
	}

	if(0 != stream.open(output, max_part_size))
	{
		qDebug() << "Fail to open stream";
		return -1;
	}

	solver->run(data, &stream, max_time, dump_step, check_step);
	return 0;
}

int main(int argc, char* argv[])
{
	QVariantMap		param(nbody_parse_arguments(argc, argv));
	nbody_data		data;
	nbcoord_t		box_size = param.value("box_size", 100).toDouble();
	QString			initial_state(param.value("initial_state", QString()).toString());

	if(initial_state.isEmpty())
	{
		size_t		stars_count = param.value("stars_count", "64").toUInt();
		data.make_universe(stars_count, box_size, box_size, box_size);
	}
	else
	{
		if(!data.load_zeno_ascii(initial_state))
		{
			qDebug() << "Can't load initial state" << initial_state;
			return -1;
		}
	}

	nbody_engine*	engine = nbody_create_engine(param);
	if(engine == NULL)
	{
		qDebug() << "Can't create engine" << param;
		return -1;
	}

	nbody_solver*	solver = nbody_create_solver(param);

	if(solver == NULL)
	{
		delete engine;
		qDebug() << "Can't create solver" << param;
		return -1;
	}

	engine->init(&data);
	solver->set_engine(engine);

	int res = con_run(argc, argv, solver, &data, param);

	delete solver;
	delete engine;

	return res;
}
