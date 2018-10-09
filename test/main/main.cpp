#include <QApplication>
#include <QDebug>
#include <omp.h>

#include "nbody_solvers.h"
#include "nbody_engines.h"
#include "nbody_data_stream.h"

QVariantMap	parse_arguments(int argc, char* argv[])
{
	QVariantMap		param;
	const QString	arg_prefix("--");

	for(int arg_n = 1; arg_n < argc; ++arg_n)
	{
		QString		arg(argv[arg_n]);
		if(!arg.startsWith(arg_prefix))
		{
			continue;
		}

		QStringList	p(arg.mid(arg_prefix.length()).split("="));
		if(p.size() != 2)
		{
			qDebug() << "Invalid argument format" << arg;
		}
		param[p[0]] = p[1];
	}

	return param;
}

int con_run(int argc, char* argv[], nbody_solver* solver, nbody_data* data, const QVariantMap& param)
{
	QString		output(param.value("output", "/tmp/nbody/main-stream").toString());
	size_t		max_part_size(static_cast<size_t>(param.value("max_part_size", 1 << 30).toULongLong()));
	nbcoord_t	max_time = param.value("max_time", 1).toDouble();
	nbcoord_t	dump_step = param.value("dump_step", 1e-2).toDouble();
	nbcoord_t	check_step = param.value("check_step", 1e-1).toDouble();

	QCoreApplication	a(argc, argv);
	nbody_data_stream	stream;

	if(0 != stream.open(output, max_part_size))
	{
		qDebug() << "Fail to open stream";
		return -1;
	}

	if(!param.value("verbose").isNull())
	{
		qDebug() << "Solver:" << solver->type_name();
		solver->print_info();
		qDebug() << "Engine:" << solver->engine()->type_name();
		solver->engine()->print_info();
	}
	solver->run(data, &stream, max_time, dump_step, check_step);
	return 0;
}

int main(int argc, char* argv[])
{
	QVariantMap		param(parse_arguments(argc, argv));
	nbody_data		data;
	nbcoord_t		box_size = param.value("box_size", 100).toDouble();
	size_t			stars_count = param.value("stars_count", "64").toUInt();

	data.make_universe(stars_count, box_size, box_size, box_size);

	nbody_engine*	engine = nbody_create_engine(param);
	if(engine == NULL)
	{
		qDebug() << "Can't create engine";
		return -1;
	}

	nbody_solver*	solver = nbody_create_solver(param);

	if(solver == NULL)
	{
		delete engine;
		qDebug() << "Can't create solver";
		return -1;
	}

	engine->init(&data);
	solver->set_engine(engine);

	int res = con_run(argc, argv, solver, &data, param);

	delete solver;
	delete engine;

	return res;
}
