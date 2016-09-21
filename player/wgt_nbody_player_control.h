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
	QLabel*				m_time;
	QPropertyAnimation*	m_animation;
	QAction*			m_act_start;
	QAction*			m_act_pause;
	QAction*			m_act_backward;
	QAction*			m_act_stop;
	QAction*			m_act_forward;
public:
	wgt_nbody_player_control( QWidget* parent, const nbody_data_stream_reader* stream );
	size_t get_current_frame() const;
public slots:
	void on_start();
	void on_pause();
	void on_backward();
	void on_stop();
	void on_forward();
	void on_finished();
signals:
	void frame_number_updated();
};

#endif // WGT_NBODY_PLAYER_CONTROL_H