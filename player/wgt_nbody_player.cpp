#include "wgt_nbody_player.h"
#include "wgt_nbody_view.h"
#include "wgt_nbody_player_control.h"

#include "nbody_data_stream_reader.h"
#include "nbody_frame_compressor_image.h"
#include "nbody_frame_compressor_opencv.h"

#include <QLayout>
#include <QDebug>
#include <QTimerEvent>
#include <QProgressDialog>
#include <QCoreApplication>
#include <QTime>

wgt_nbody_player::wgt_nbody_player(nbody_solver* solver, nbody_data* _data, nbcoord_t box_size)
{
	QVBoxLayout*	layout = new QVBoxLayout(this);

	setAttribute(Qt::WA_DeleteOnClose);
	setMinimumSize(320, 240);

	m_solver = solver;
	m_data = _data;
	m_view = new wgt_nbody_view(solver, _data, box_size);
	m_stream = new nbody_data_stream_reader();
	m_stream->load("/home/sas/Documents/tmp/nbody/main-stream");
	m_data->resize(m_stream->get_body_count());
	qDebug() << "Load stream" << m_stream->get_max_time();
	m_control = new wgt_nbody_player_control(this, m_stream);
	layout->addWidget(m_view, 1000);
	layout->addWidget(m_control);

	connect(m_control, SIGNAL(frame_number_updated()),
			this, SLOT(on_update_data()));
	connect(m_control, SIGNAL(frame_state_updated()),
			this, SLOT(on_update_view()));
	connect(m_control, SIGNAL(star_intensity_updated()),
			this, SLOT(on_update_view()));
	connect(m_control, SIGNAL(star_size_updated()),
			this, SLOT(on_update_view()));
	connect(m_control, SIGNAL(start_record()),
			this, SLOT(on_start_record()));
	connect(m_view, SIGNAL(stars_size_range_changed(double, double, double)),
			m_control, SLOT(on_stars_size_range_changed(double, double, double)));
}

wgt_nbody_player::~wgt_nbody_player()
{
	delete m_stream;
}

void wgt_nbody_player::on_update_data()
{
	if(0 != m_stream->seek(m_control->get_current_frame()))
	{
		return;
	}

	if(0 != m_stream->read(m_data))
	{
		return;
	}

	on_update_view();
}

void wgt_nbody_player::on_update_view()
{
	m_view->set_stereo_base(m_control->get_stereo_base());
	m_view->set_star_intensity(m_control->get_star_intensity());
	m_view->set_star_size(m_control->get_star_size());
	m_view->updateGL();
}

void wgt_nbody_player::on_start_record()
{
	QProgressDialog		progress(this);
	QTime				timer;
	QString				out_dir("/home/sas/Documents/tmp/nbody/video.avi");

	progress.setRange(0, static_cast<int>(m_stream->get_frame_count()));
	progress.show();
	timer.start();

	nbody_frame_compressor_opencv	compressor;

	if(!compressor.set_destination(out_dir))
	{
		qDebug() << "can't setup compressor";
		return;
	}

	size_t	frame_count = m_stream->get_frame_count();

	for(size_t frame_n = 0; frame_n != frame_count; ++frame_n)
	{
		if(0 != m_stream->seek(frame_n))
		{
			qDebug() << "Fail to seek stream frame #" << frame_n;
			break;
		}

		if(0 != m_stream->read(m_data))
		{
			qDebug() << "Fail to read stream frame #" << frame_n;
			break;
		}

		if(frame_n % 100 == 0)
		{
			m_data->print_statistics(NULL);
		}

		QImage	frame(m_view->render_to_image());

		if(frame.isNull())
		{
			qDebug() << "Render frame failed";
			break;
		}

		compressor.push_frame(frame, frame_n);

		progress.setValue(static_cast<int>(frame_n));
		progress.setLabelText(QString("Done %1 from %2 ( %3 fps )")
							  .arg(frame_n)
							  .arg(m_stream->get_frame_count())
							  .arg(static_cast<double>(frame_n) / (timer.elapsed() / 1000.0)));
		if(progress.wasCanceled())
		{
			break;
		}
		QCoreApplication::processEvents();
	}

	on_update_data();

	qDebug() << "Record done!";
}
