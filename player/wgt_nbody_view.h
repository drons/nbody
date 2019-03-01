#ifndef WGT_NBODY_VIEW_H
#define WGT_NBODY_VIEW_H

#include <QGLWidget>
#include <QGLFramebufferObject>
#include "nbody_solver.h"

class wgt_nbody_view : public QGLWidget
{
	Q_OBJECT
	Q_PROPERTY(QPointF m_split_point READ get_split_point WRITE set_split_point)
	nbody_data*				m_data;
	nb3d_t					m_box_min;
	nb3d_t					m_box_max;
	nb3d_t					m_vel_min;
	nb3d_t					m_vel_max;
	QGLFramebufferObject*	m_renderer;
	QPointF					m_split_point;
	int						m_stereo_base;
	int						m_star_intensity;
	double					m_star_size;
	bool					m_color_from_velosity;
	bool					m_show_box;
public:
	explicit wgt_nbody_view(nbody_data*);
	~wgt_nbody_view();

	QPointF get_split_point() const;
	void set_split_point(const QPointF& split_point);
	void set_stereo_base(int);
	void set_star_intensity(int);
	void set_star_size(double);
	void paint_color_box();
	void initializeGL() override;
	void paintGL() override;
	void paintGL(GLsizei width, GLsizei height);
	void paintGL(GLint x, GLint y,	GLsizei width, GLsizei height, const nbvertex_t& camera_position, const nbvertex_t& up);
	void setup_view_box();
	void setup_projection(GLsizei width, GLsizei height, const nbvertex_t& center, const nbvertex_t& camera_position,
						  const nbvertex_t& up);
	QImage render_to_image();
	void mouseDoubleClickEvent(QMouseEvent*) override;
	bool get_color_from_velosity() const;
	void set_color_from_velosity(bool color_from_velosity);
	void set_show_box(bool show_box);
signals:
	void stars_size_range_changed(double, double, double);
};


#endif // WGT_NBODY_VIEW_H
