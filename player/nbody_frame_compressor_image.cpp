#include "nbody_frame_compressor_image.h"

#include <QDir>
#include <QDebug>

nbody_frame_compressor_image::nbody_frame_compressor_image()
{

}

nbody_frame_compressor_image::~nbody_frame_compressor_image()
{
	wait_results(0);
}

bool nbody_frame_compressor_image::set_destination(const QString& d)
{
	if(!QDir(d).mkpath("."))
	{
		qDebug() << "Can't mkpath" << d;
		return false;
	}

	m_destination = d;

	m_name_tmpl = d + "/%1.png";

	return true;
}

static bool frame_writer(const QImage& img, const QString fname)
{
	return img.save(fname, "PNG");
}

void nbody_frame_compressor_image::push_frame(const QImage& frame, size_t frame_n)
{
	QString		frame_name(m_name_tmpl.arg(frame_n, 8, 10,  QChar('0')));

	m_results << QtConcurrent::run(frame_writer, frame, frame_name);

	wait_results(QThread::idealThreadCount());
}

void nbody_frame_compressor_image::wait_results(int max_queue_size)
{
	while(m_results.size() > max_queue_size)
	{
		for(QList< QFuture<bool> >::iterator ii = m_results.begin(); ii != m_results.end(); ++ii)
		{
			if(ii->isFinished())
			{
				m_results.erase(ii);
				break;
			}
		}
	}
}
