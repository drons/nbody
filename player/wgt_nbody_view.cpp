#include "wgt_nbody_view.h"
#include "nbody_solver.h"

#include <GL/glu.h>
#include <omp.h>
#include <QDebug>
#include <QMouseEvent>
#include <QPropertyAnimation>

wgt_nbody_view::wgt_nbody_view(nbody_solver* solver, nbody_data* _data, nbcoord_t box_size)
{
	setAttribute(Qt::WA_DeleteOnClose);

	m_split_point = QPointF(0.5, 0.5);
	m_mesh_sx = box_size;
	m_mesh_sy = box_size;
	m_mesh_sz = box_size;

	m_data = _data;
	m_solver = solver;
	m_renderer = NULL;
	m_stereo_base = 0;
	m_star_intensity = 255;
	m_star_size = 1;
}

wgt_nbody_view::~wgt_nbody_view()
{
	delete m_renderer;
}

void wgt_nbody_view::paint_color_box()
{
	glEnable(GL_DEPTH_TEST);
	glShadeModel(GL_SMOOTH);
	//! Draw Cube
	glLineWidth(1);
	const nb3d_t	cube_vrt[8] =
	{
		//Floor
		nb3d_t(0, 0, 0),
		nb3d_t(m_mesh_sx, 0, 0),
		nb3d_t(m_mesh_sx, m_mesh_sy, 0),
		nb3d_t(0, m_mesh_sy, 0),
		//Roof
		nb3d_t(0, 0, m_mesh_sz),
		nb3d_t(m_mesh_sx, 0, m_mesh_sz),
		nb3d_t(m_mesh_sx, m_mesh_sy, m_mesh_sz),
		nb3d_t(0, m_mesh_sy, m_mesh_sz)
	};

	const nb3d_t	cube_col[8] =
	{
		//Floor
		nb3d_t(0, 0, 0),
		nb3d_t(1, 0, 0),
		nb3d_t(1, 1, 0),
		nb3d_t(0, 1, 0),
		//Roof
		nb3d_t(0, 0, 1),
		nb3d_t(1, 0, 1),
		nb3d_t(1, 1, 1),
		nb3d_t(0, 1, 1)
	};
	const GLuint cube_idx[12 * 2] =
	{
		//Floor
		0, 1,
		1, 2,
		2, 3,
		3, 0,
		//Roof
		4, 5,
		5, 6,
		6, 7,
		7, 4,
		//Wall
		0, 4,
		1, 5,
		2, 6,
		3, 7
	};

	glBegin(GL_LINES);

	for(int i = 0; i != 12; ++i)
	{
		int	idx1 = cube_idx[ 2 * i ];
		int	idx2 = cube_idx[ 2 * i + 1 ];

		glColor3dv(cube_col[ idx1 ].data());
		glVertex3dv(cube_vrt[ idx1 ].data());

		glColor3dv(cube_col[ idx2 ].data());
		glVertex3dv(cube_vrt[ idx2 ].data());
	}

	glEnd();

	for(int i = 0; i != 8; ++i)
	{
		renderText(cube_vrt[i].x, cube_vrt[i].y, cube_vrt[i].z, QString::number(i));
	}
}

void wgt_nbody_view::initializeGL()
{
	m_renderer = new QGLFramebufferObject(1920, 1080);

	GLfloat size_range[2] = {1, 1};
	GLfloat size_step = 1;

	glGetFloatv(GL_POINT_SIZE_RANGE, size_range);
	glGetFloatv(GL_POINT_SIZE_GRANULARITY, &size_step);

	emit stars_size_range_changed(size_range[0], size_range[1], size_step);
}

void wgt_nbody_view::paintGL(GLint x, GLint y, GLsizei width, GLsizei height, const nbvertex_t& camera_position,
							 const nbvertex_t& up)
{
	glViewport(x, y, width, height);

	glColor3f(1, 1, 1);
	renderText(20 , 20, QString("Step  = %1").arg(m_data->get_step()), QFont("Monospace"));
	renderText(20 , 40, QString("T     = %1").arg(m_data->get_time()), QFont("Monospace"));
	renderText(20 , 60, QString("Stars = %1").arg(m_data->get_count()), QFont("Monospace"));
	renderText(20 , 80, QString("dP    = %1 %").arg(m_data->get_impulce_err(), 3, 'e', 2), QFont("Monospace"));
	renderText(20 , 100, QString("dL    = %1 %").arg(m_data->get_impulce_moment_err(), 3, 'e', 2), QFont("Monospace"));
	renderText(20 , 120, QString("dE    = %1 %").arg(m_data->get_energy_err(), 3, 'e', 2), QFont("Monospace"));


	glDisable(GL_DEPTH_TEST);
	glLineWidth(1);
	glPointSize(static_cast<GLfloat>(m_star_size));
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_BLEND);

	nbvertex_t	center(m_mesh_sx * 0.5, m_mesh_sy * 0.5, m_mesh_sz * 0.5);
	GLfloat		factor(static_cast<GLfloat>(m_star_intensity) / 255.0f);

	if(m_stereo_base == 0)
	{
		setup_projection(width, height, center, camera_position, up);
		paint_color_box();

		glBlendFunc(GL_CONSTANT_ALPHA, GL_ONE);
		glBlendColor(factor, factor, factor, factor);

		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);
		glVertexPointer(nbtype_info<nbvertex_t>::size(), nbtype_info<nbvertex_t>::gl_type(), 0, m_data->get_vertites());
		glColorPointer(nbtype_info<nbcolor_t>::size(), nbtype_info<nbcolor_t>::gl_type(), 0, m_data->get_color());
		glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(m_data->get_count()));
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);
	}
	else
	{
		nbvertex_t	camera_ray(center - camera_position);
		nbvertex_t	base(camera_ray ^ up);


		base *= (camera_ray.length() / base.length());
		base *= (m_stereo_base / 1000.0);
		nbcolor_t	col[] = { nbcolor_t(1,  0, 0, 1),
							  nbcolor_t(0,  1, 1, 1)
						  };
		nbvertex_t	cpos[] = { camera_position + base,
							   camera_position - base
							};

		glBlendFunc(GL_ONE, GL_ONE);

		for(size_t plane = 0; plane != 2; ++plane)
		{
			setup_projection(width, height, center, cpos[plane], up);
			//paint_color_box();
			glColor3f(col[plane].x * factor, col[plane].y * factor, col[plane].z * factor);
			glEnableClientState(GL_VERTEX_ARRAY);
			glVertexPointer(nbtype_info<nbvertex_t>::size(), nbtype_info<nbvertex_t>::gl_type(), 0, m_data->get_vertites());
			glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(m_data->get_count()));
			glDisableClientState(GL_VERTEX_ARRAY);
		}
	}

	glBlendFunc(GL_ONE, GL_ONE);
}

void wgt_nbody_view::setup_projection(GLsizei width, GLsizei height, const nbvertex_t& center,
									  const nbvertex_t& camera_position, const nbvertex_t& up)
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	GLfloat aspect = static_cast<GLfloat>(height) / static_cast<GLfloat>(width);
	GLfloat	near = 1;
	GLfloat	far = 1000;

	gluPerspective(60, 1.0 / aspect, near, far);
	gluLookAt(camera_position.x, camera_position.y, camera_position.z,
			  center.x, center.y, center.z,
			  up.x, up.y, up.z);
}

QImage wgt_nbody_view::render_to_image()
{
	makeCurrent();

	if(!m_renderer->bind())
	{
		qDebug() << "Can't bind QGLFramebufferObject";
		return QImage();
	}
	paintGL(m_renderer->width(), m_renderer->height());
	if(!m_renderer->release())
	{
		qDebug() << "Can't release QGLFramebufferObject";
		return QImage();
	}
	QImage	image(m_renderer->toImage());
	if(image.isNull())
	{
		qDebug() << "Can't convert QGLFramebufferObject to image";
		return QImage();
	}

	{
		QPainter	p(&image);
		p.setRenderHints(QPainter::Antialiasing | QPainter::TextAntialiasing | QPainter::HighQualityAntialiasing);
		p.setPen(Qt::white);
		p.setFont(QFont("Monospace", 8));
		p.drawLine(QPointF(0, image.height() / 2.0), QPointF(image.width(), image.height() / 2.0));
		p.drawLine(QPointF(image.width() / 2.0, 0), QPointF(image.width() / 2.0, image.height()));

		p.drawText(20 , 20, QString("Step  = %1").arg(m_data->get_step()));
		p.drawText(20 , 40, QString("T     = %1").arg(m_data->get_time()));
		p.drawText(20 , 60, QString("Stars = %1").arg(m_data->get_count()));
		p.drawText(20 , 80, QString("dP    = %1 %").arg(m_data->get_impulce_err(), 3, 'e', 2));
		p.drawText(20 , 100, QString("dL    = %1 %").arg(m_data->get_impulce_moment_err(), 3, 'e', 2));
		p.drawText(20 , 120, QString("dE    = %1 %").arg(m_data->get_energy_err(), 3, 'e', 2));
	}
	return image;
}

void wgt_nbody_view::paintGL(GLsizei width, GLsizei height)
{
	glViewport(0, 0, width, height);
	qglClearColor(Qt::black);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	nbvertex_t	center(m_mesh_sx * 0.5, m_mesh_sy * 0.5, m_mesh_sz * 0.5);
	nbcoord_t	dist = 200;
	GLint		x = static_cast<GLint>(width * m_split_point.x());
	GLint		y = static_cast<GLint>(height * m_split_point.y());

	paintGL(x, y, width - x, height - y, center - nbvertex_t(0, dist, dist), nbvertex_t(0, 0, 1));
	paintGL(0, 0, x, y, center - nbvertex_t(0, 0, dist), nbvertex_t(0, 1, 0));
	paintGL(x, 0, width - x, y, center + nbvertex_t(dist, 0, 0), nbvertex_t(0, 1, 0));
	paintGL(0, y, x, height - y, center - nbvertex_t(0, dist, 0), nbvertex_t(0, 0, -1));
}

void wgt_nbody_view::paintGL()
{
	m_solver->engine()->get_data(m_data);
	paintGL(width(), height());
}

void wgt_nbody_view::step()
{
	size_t	i = m_data->get_step();

	size_t	w = 100;
	if(i % w == 0)
	{
		m_data->print_statistics(m_solver->engine());
		//render_file();
	}

	nbcoord_t	step_time = omp_get_wtime();
	m_solver->advise(m_solver->get_max_step());
	qDebug() << "Step time" << step_time - omp_get_wtime();
}

void wgt_nbody_view::mouseDoubleClickEvent(QMouseEvent* e)
{
	QPoint	p(e->pos());
	QPoint	s(size().width() / 2, size().height() / 2);
	QPointF	new_split(m_split_point);

	if(m_split_point == QPointF(0.5, 0.5))
	{
		if(p.x() > s.x() && p.y() > s.y())
		{
			new_split = QPointF(0, 1);
		}
		else if(p.x() > s.x() && p.y() < s.y())
		{
			new_split = QPointF(0, 0);
		}
		else if(p.x() < s.x() && p.y() > s.y())
		{
			new_split = QPointF(1, 1);
		}
		else if(p.x() < s.x() && p.y() < s.y())
		{
			new_split = QPointF(1, 0);
		}
	}
	else
	{
		new_split = QPointF(0.5, 0.5);
	}

	QPropertyAnimation*	anim = new QPropertyAnimation(this, "m_split_point");
	anim->setStartValue(m_split_point);
	anim->setEndValue(new_split);
	anim->setDuration(500);
	anim->start(QAbstractAnimation::DeleteWhenStopped);
}

QPointF wgt_nbody_view::get_split_point() const
{
	return m_split_point;
}

void wgt_nbody_view::set_split_point(const QPointF& split_point)
{
	m_split_point = split_point;
	updateGL();
}

void wgt_nbody_view::set_stereo_base(int base)
{
	m_stereo_base = base;
}

void wgt_nbody_view::set_star_intensity(int star_intensity)
{
	m_star_intensity = star_intensity;
}

void wgt_nbody_view::set_star_size(double star_size)
{
	m_star_size = star_size;
}
