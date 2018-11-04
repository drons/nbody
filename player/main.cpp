#include <QApplication>
#include <QDebug>
#include <omp.h>

#include "wgt_nbody_player.h"
#include "nbody_arg_parser.h"
#include "nbody_data_stream_reader.h"
#include "nbody_solvers.h"
#include "nbody_engines.h"


int main(int argc, char* argv[])
{
	QApplication				app(argc, argv);
	QVariantMap					param(nbody_parse_arguments(argc, argv));
	QString						stream_name(param.value("input", "/tmp/nbody/main-stream").toString());
	nbody_data_stream_reader	stream;

	if(0 != stream.load(stream_name))
	{
		qDebug() << "can't open stream" << stream_name;
		return -1;
	}

	qDebug() << "Stream name:" << stream_name;
	qDebug() << "Star count: " << stream.get_body_count();
	qDebug() << "Frame count:" << stream.get_frame_count();

	wgt_nbody_player*	nbv = new wgt_nbody_player(&stream);

	nbv->show();

	return app.exec();
}
