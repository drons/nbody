#ifndef WGT_NBODY_VIEW_H
#define WGT_NBODY_VIEW_H

#include <QGLWidget>
#include <QGLFramebufferObject>
#include "nbody_solver.h"

class wgt_nbody_view : public QGLWidget
{
	Q_OBJECT
	Q_PROPERTY( QPointF m_split_point READ get_split_point WRITE set_split_point );
	nbody_data*				m_data;
	nbody_solver*			m_solver;
	nbcoord_t				m_mesh_sx;
	nbcoord_t				m_mesh_sy;
	nbcoord_t				m_mesh_sz;
	QGLFramebufferObject*	m_renderer;
	QPointF					m_split_point;
	int						m_stereo_base;
	int						m_star_intensity;
public:
	wgt_nbody_view( nbody_solver*, nbody_data*, nbcoord_t box_size );
	~wgt_nbody_view();

	QPointF get_split_point() const;
	void set_split_point( const QPointF& split_point );
	void set_stereo_base( int );
	void set_star_intensity( int );
	void paint_color_box();
	void initializeGL();
	void paintGL();
	void paintGL( GLsizei width, GLsizei height );
	void paintGL( GLint x, GLint y,	GLsizei width, GLsizei height, const nbvertex_t& camera_position, const nbvertex_t& up );
	void setup_projection( GLsizei width, GLsizei height, const nbvertex_t& center, const nbvertex_t& camera_position, const nbvertex_t& up );
	void render_file();
	void step();
	void mouseDoubleClickEvent( QMouseEvent* );

};


#endif // WGT_NBODY_VIEW_H
