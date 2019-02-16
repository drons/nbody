#include "wgt_nbody_player.h"
#include "wgt_nbody_view.h"
#include "wgt_nbody_player_control.h"

#include "nbody_data_stream_reader.h"
#include "nbody_frame_compressor_image.h"
#include "nbody_frame_compressor_opencv.h"

#include <memory>
#include <QLayout>
#include <QDebug>
#include <QTimerEvent>
#include <QProgressDialog>
#include <QCoreApplication>
#include <QTime>
#include <QFileDialog>

wgt_nbody_player::wgt_nbody_player(nbody_data_stream_reader* stream,
								   const QString& check_list)
{
	QVBoxLayout*	layout = new QVBoxLayout(this);

	setAttribute(Qt::WA_DeleteOnClose);
	setMinimumSize(320, 240);

	m_data = new nbody_data;
	m_data->set_check_list(check_list);
	m_stream = stream;
	m_view = new wgt_nbody_view(m_data);
	m_data->resize(m_stream->get_body_count());
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
	connect(m_control, SIGNAL(color_from_velosity_changed()),
			this, SLOT(on_update_view()));
	connect(m_view, SIGNAL(stars_size_range_changed(double, double, double)),
			m_control, SLOT(on_stars_size_range_changed(double, double, double)));
}

wgt_nbody_player::~wgt_nbody_player()
{
	delete m_data;
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
	m_view->set_color_from_velosity(m_control->get_color_from_velosity());
	m_view->updateGL();
}

void wgt_nbody_player::on_start_record()
{
	QStringList		filters{"Avi (*.avi)", "PNG frames (*.png)"};
	QString			selected;
	QString			out(QFileDialog::getSaveFileName(this, "Select output video stream",
													 QDir::homePath(), filters.join("\n"),
													 &selected));

	std::shared_ptr<nbody_frame_compressor>	compressor;
	if(selected == filters[0])
	{
		compressor = std::make_shared<nbody_frame_compressor_opencv>();
		if(!out.endsWith(".avi", Qt::CaseInsensitive))
		{
			out += ".avi";
		}
	}
	else if(selected == filters[1])
	{
		compressor = std::make_shared<nbody_frame_compressor_image>();
	}

	if((compressor == NULL) || (!compressor->set_destination(out)))
	{
		qDebug() << "can't setup compressor" << out;
		return;
	}

	QProgressDialog	progress(this);
	QTime			timer;

	progress.setRange(0, static_cast<int>(m_stream->get_frame_count()));
	progress.show();
	timer.start();

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

		compressor->push_frame(frame, frame_n);

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
