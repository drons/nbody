#ifndef WGT_NBODY_VIEW_H
#define WGT_NBODY_VIEW_H

#include <QGLWidget>
#include <QGLFramebufferObject>
#include "nbody_solver.h"

class wgt_nbody_view : public QGLWidget
{
	nbody_data				m_3body;
	nbody_solver*			m_solver;
	nbody_fcompute*			m_engine;
	nbcoord_t				m_mesh_sx;
	nbcoord_t				m_mesh_sy;
	nbcoord_t				m_mesh_sz;
	QGLFramebufferObject*	m_renderer;
public:
	wgt_nbody_view();
	~wgt_nbody_view();

	void paint_color_box();
	void initializeGL();
	void paintGL();
	void paintGL( GLsizei width, GLsizei height );
	void paintGL( GLint x, GLint y,	GLsizei width, GLsizei height, const nbvertex_t& camera_position, const nbvertex_t& up );
	void render_file();
	void step();
};


#endif // WGT_NBODY_VIEW_H
