#ifndef WGT_NBODY_PLAYER_CONTROL_H
#define WGT_NBODY_PLAYER_CONTROL_H

#include <QWidget>
#include <QPropertyAnimation>

class nbody_data_stream_reader;
class QLabel;
class QSlider;

class wgt_nbody_player_control : public QWidget
{
	Q_OBJECT
	QSlider*			m_timeline;
	QSlider*			m_stereo_base;
	QLabel*				m_frame_number;
	QPropertyAnimation*	m_animation;
	QAction*			m_act_start;
	QAction*			m_act_pause;
	QAction*			m_act_backward;
	QAction*			m_act_stop;
	QAction*			m_act_forward;
public:
	wgt_nbody_player_control( QWidget* parent, const nbody_data_stream_reader* stream );
	size_t get_current_frame() const;
	int get_stereo_base() const;
public slots:
	void on_start();
	void on_pause();
	void on_backward();
	void on_stop();
	void on_forward();
	void on_finished();
	void on_stereo_base_changed( int );
	void on_frame_number_updated();
signals:
	void frame_number_updated();
	void frame_state_updated();
};

#endif // WGT_NBODY_PLAYER_CONTROL_H
