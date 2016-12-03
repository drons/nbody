#include "wgt_nbody_player.h"
#include "wgt_nbody_view.h"
#include "wgt_nbody_player_control.h"

#include "nbody_data_stream_reader.h"

#include <QLayout>
#include <QDebug>
#include <QTimerEvent>

wgt_nbody_player::wgt_nbody_player( nbody_solver* solver, nbody_data* data, nbcoord_t box_size )
{
	QVBoxLayout*	layout = new QVBoxLayout( this );

	setAttribute( Qt::WA_DeleteOnClose );
	setMinimumSize( 320, 240 );

	m_solver = solver;
	m_data = data;
	m_view = new wgt_nbody_view( solver, data, box_size );
	m_stream = new nbody_data_stream_reader();
	m_stream->load( "/home/sas/tmp/nbody/main-stream" );
	qDebug() << "Load stream" << m_stream->get_max_time();
	m_control = new wgt_nbody_player_control( this, m_stream );
	layout->addWidget( m_view, 1000 );
	layout->addWidget( m_control );

	connect( m_control, SIGNAL( frame_number_updated() ),
			 this, SLOT( on_update_data() ) );
	connect( m_control, SIGNAL( frame_state_updated() ),
			 this, SLOT( on_update_view() ) );
	connect( m_control, SIGNAL( star_intensity_updated() ),
			 this, SLOT( on_update_view() ) );
}

wgt_nbody_player::~wgt_nbody_player()
{
	delete m_stream;
}

void wgt_nbody_player::on_update_data()
{
	if( 0 != m_stream->seek( m_control->get_current_frame() ) )
	{
		return;
	}

	if( 0 != m_stream->read( m_solver->engine() ) )
	{
		return;
	}

//	m_data->print_statistics( m_solver->engine() );

	m_solver->engine()->get_data( m_data );
	on_update_view();
}

void wgt_nbody_player::on_update_view()
{
	m_view->set_stereo_base( m_control->get_stereo_base() );
	m_view->set_star_intensity( m_control->get_star_intensity() );
	m_view->updateGL();
}
