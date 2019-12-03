#include <QCoreApplication>
#include <QDebug>
#include <memory>
#include <omp.h>

#include "nbody_solvers.h"
#include "nbody_engines.h"
#include "nbody_data_stream.h"
#include "nbody_data_stream_reader.h"
#include "nbody_arg_parser.h"

int main(int argc, char* argv[])
{
	QCoreApplication	a(argc, argv);
	QVariantMap			param(nbody_parse_arguments(argc, argv));
	nbcoord_t			box_size(param.value("box_size", 100).toDouble());
	size_t				max_part_size(static_cast<size_t>(param.value("max_part_size", 1 << 30).toULongLong()));
	nbcoord_t			max_time = param.value("max_time", 1).toDouble();
	nbcoord_t			dump_step = param.value("dump_step", 1e-2).toDouble();
	nbcoord_t			check_step = param.value("check_step", 1e-1).toDouble();
	QString				check_list(param.value("check_list", "PLV").toString());
	QString				output(param.value("output", "/tmp/nbody/main-stream").toString());
	QString				resume(param.value("resume", QString()).toString());
	QString				initial_state(param.value("initial_state", QString()).toString());
	QString				initial_state_type(param.value("initial_type", "Zeno").toString());

	nbody_data									data;
	std::unique_ptr<nbody_data_stream>			stream(new nbody_data_stream);
	std::unique_ptr<nbody_data_stream_reader>	resume_stream;

	if(!resume.isEmpty())
	{
		resume_stream.reset(new nbody_data_stream_reader);
		if(0 != resume_stream->load(resume))
		{
			qDebug() << "Can't open stream to resume" << resume;
			return -1;
		}
		if(0 != resume_stream->seek(resume_stream->get_frame_count() - 1))
		{
			qDebug() << "Can't seek to the end of stream" << resume;
			return -1;
		}
		data.resize(resume_stream->get_body_count());
		if(0 != resume_stream->read(&data))
		{
			qDebug() << "Can't read last frame from stream" << resume;
			return -1;
		}
		output = resume;
	}
	else if(!initial_state.isEmpty())
	{
		if(!data.load_initial(initial_state, initial_state_type))
		{
			qDebug() << "Can't load initial state" << initial_state;
			return -1;
		}
	}
	else
	{
		size_t		stars_count = param.value("stars_count", "64").toUInt();
		data.make_universe(stars_count, box_size, box_size, box_size);
	}

	std::unique_ptr<nbody_engine>	engine(nbody_create_engine(param));
	if(engine == NULL)
	{
		qDebug() << "Can't create engine" << param;
		return -1;
	}

	std::unique_ptr<nbody_solver>	solver(nbody_create_solver(param));
	if(solver == NULL)
	{
		qDebug() << "Can't create solver" << param;
		return -1;
	}

	engine->init(&data);
	solver->set_engine(engine.get());

	data.set_check_list(check_list);

	if(param.value("verbose", "0").toInt() != 0)
	{
		qDebug() << "General:";
		qDebug() << "\tStars count:" << data.get_count();
		qDebug() << "\tBox size:" << data.get_box_size();
		qDebug() << "\toutput:" << output;
		if(!resume.isEmpty())
		{
			qDebug() << "\tresume:" << resume;
		}
		qDebug() << "\tinitial_state:" << initial_state;
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

	if(0 != stream->open(output, max_part_size, resume_stream.get()))
	{
		qDebug() << "Fail to open stream";
		return -1;
	}

	if(resume_stream != NULL)
	{
		resume_stream->close();
	}

	return solver->run(&data, stream.get(), max_time, dump_step, check_step);;
}
