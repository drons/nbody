#include "wgt_nbody_player_control.h"
#include "nbody_data_stream_reader.h"

#include <QSlider>
#include <QLabel>
#include <QLayout>
#include <QToolBar>
#include <QAction>
#include <QIcon>
#include <QDebug>

wgt_nbody_player_control::wgt_nbody_player_control( QWidget* parent, const nbody_data_stream_reader* stream ) :
	QWidget( parent )
{
	QHBoxLayout*	layout = new QHBoxLayout( this );
	QToolBar*		bar = new QToolBar( this );

	QIcon::setThemeName( "nuvola" );
	m_act_start = bar->addAction( QIcon::fromTheme( "media-playback-start" ), tr( "Start" ) );
	m_act_pause = bar->addAction( QIcon::fromTheme( "media-playback-pause" ), tr( "Pause" ) );
	m_act_backward = bar->addAction( QIcon::fromTheme( "media-seek-backward" ), tr( "<Backward" ) );
	m_act_stop = bar->addAction( QIcon::fromTheme( "media-playback-stop" ), tr( "Stop" ) );
	m_act_forward = bar->addAction( QIcon::fromTheme( "media-seek-forward" ), tr( "Forward>" ) );

	m_timeline = new QSlider( this );
	m_stereo_base = new QSlider( this );
	m_timeline->setOrientation( Qt::Horizontal );
	m_stereo_base->setOrientation( Qt::Horizontal );
	layout->addWidget( bar );
	layout->addWidget( m_timeline );
	layout->addWidget( m_stereo_base );

	m_timeline->setRange( 0, (int)stream->get_frame_count() - 1 );
	m_stereo_base->setRange( 0, 100 );

	m_animation = new QPropertyAnimation( this );
	m_animation->setDuration( 5000 );
	m_animation->setStartValue( m_timeline->minimum() );
	m_animation->setEndValue( m_timeline->maximum() );
	m_animation->setEasingCurve( QEasingCurve::Linear );
	m_animation->setPropertyName( "value" );
	m_animation->setTargetObject( m_timeline );

	connect( m_timeline, SIGNAL( sliderMoved(int) ),
			this, SIGNAL( frame_number_updated() ) );
	connect( m_timeline, SIGNAL( valueChanged(int) ),
			this, SIGNAL( frame_number_updated() ) );
	connect( m_animation, SIGNAL( finished() ),
			 this, SLOT( on_finished() ) );
	connect( m_stereo_base, SIGNAL( valueChanged(int) ),
			 this, SLOT( on_stereo_base_changed(int) ) );
	connect( m_act_start, SIGNAL( triggered(bool) ), this, SLOT( on_start() ) );
	connect( m_act_pause, SIGNAL( triggered(bool) ), this, SLOT( on_pause() ) );
	connect( m_act_backward, SIGNAL( triggered(bool) ), this, SLOT( on_backward() ) );
	connect( m_act_stop, SIGNAL( triggered(bool) ), this, SLOT( on_stop() ) );
	connect( m_act_forward, SIGNAL( triggered(bool) ), this, SLOT( on_forward() ) );

	on_finished();
}

size_t wgt_nbody_player_control::get_current_frame() const
{
	return (size_t)m_timeline->value();
}

int wgt_nbody_player_control::get_stereo_base() const
{
	return m_stereo_base->value();
}

void wgt_nbody_player_control::on_start()
{
	m_animation->start();
	m_act_start->setEnabled( false );
	m_act_stop->setEnabled( true );
	m_act_pause->setEnabled( true );
	m_act_backward->setEnabled( true );
	m_act_forward->setEnabled( true );
}

void wgt_nbody_player_control::on_pause()
{
	m_animation->pause();
	m_act_pause->setEnabled( false );
	m_act_start->setEnabled( true );
}

void wgt_nbody_player_control::on_backward()
{
	m_animation->setCurrentTime(0);
	m_timeline->setValue( m_timeline->minimum() );
}

void wgt_nbody_player_control::on_stop()
{
	m_animation->stop();
	m_animation->setCurrentTime(0);
	m_timeline->setValue( m_timeline->minimum() );
	on_finished();
}

void wgt_nbody_player_control::on_forward()
{
	m_animation->setCurrentTime( m_animation->duration() );
	m_timeline->setValue( m_timeline->maximum() );
}

void wgt_nbody_player_control::on_finished()
{
	m_act_start->setEnabled( true );
	m_act_stop->setEnabled( false );
	m_act_pause->setEnabled( false );
}

void wgt_nbody_player_control::on_stereo_base_changed( int )
{
	emit frame_state_updated();
}
