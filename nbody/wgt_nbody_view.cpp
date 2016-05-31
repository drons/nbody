#include "wgt_nbody_view.h"

#include "nbody_solver_adams.h"
#include "nbody_solver_euler.h"
#include "nbody_solver_runge_kutta.h"
#include "nbody_solver_stormer.h"

#include <GL/glu.h>
#include <omp.h>
#include <QDebug>
#include <QDir>

wgt_nbody_view::wgt_nbody_view()
{
	setAttribute( Qt::WA_DeleteOnClose );

	m_mesh_sx = 100;
	m_mesh_sy = 100;
	m_mesh_sz = 100;
	nbcoord_t	radius = 50;
	nbcoord_t	galaxy_mass = 1000;
	size_t		star_count = 30;
	nbvertex_t	center( m_mesh_sx*0.5, m_mesh_sy*0.5, m_mesh_sz*0.5 );
	nbvertex_t	base( radius, 0, 0 );
	nbvertex_t	velosity( 0, sqrt(m_3body.force( nbvertex_t(), base, galaxy_mass, galaxy_mass ).length()*(base).length()/(2*galaxy_mass)), 0 );
	srand(1);

	m_3body.add_galaxy( center - base, velosity/3, radius, galaxy_mass, star_count );
	m_3body.add_galaxy( center + base, -velosity/3, radius, galaxy_mass, star_count );
	//m_3body.add_galaxy( center, vertex_t(), radius, galaxy_mass, star_count );

	m_solver = new nbody_solver_runge_kutta( &m_3body );
	m_renderer = NULL;
}

wgt_nbody_view::~wgt_nbody_view()
{
	delete m_solver;
	delete m_renderer;
}

void wgt_nbody_view::paint_color_box()
{
	glEnable( GL_DEPTH_TEST );
	glShadeModel( GL_SMOOTH );
	//! Draw Cube
	glLineWidth( 1 );
	const nbvertex_t	cube_vrt[8] =
	{
		//Floor
		nbvertex_t( 0, 0, 0 ),
		nbvertex_t( m_mesh_sx, 0, 0 ),
		nbvertex_t( m_mesh_sx, m_mesh_sy, 0 ),
		nbvertex_t( 0, m_mesh_sy, 0 ),
		//Roof
		nbvertex_t( 0, 0, m_mesh_sz ),
		nbvertex_t( m_mesh_sx, 0, m_mesh_sz ),
		nbvertex_t( m_mesh_sx, m_mesh_sy, m_mesh_sz ),
		nbvertex_t( 0, m_mesh_sy, m_mesh_sz )
	};

	const nbvertex_t	cube_col[8] =
	{
		//Floor
		nbvertex_t( 0, 0, 0 ),
		nbvertex_t( 1, 0, 0 ),
		nbvertex_t( 1, 1, 0 ),
		nbvertex_t( 0, 1, 0 ),
		//Roof
		nbvertex_t( 0, 0, 1 ),
		nbvertex_t( 1, 0, 1 ),
		nbvertex_t( 1, 1, 1 ),
		nbvertex_t( 0, 1, 1 )
	};
	const GLuint cube_idx[12*2] =
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

	glBegin( GL_LINES );

	for( int i = 0; i != 12; ++i )
	{
		int	idx1 = cube_idx[ 2*i ];
		int	idx2 = cube_idx[ 2*i + 1 ];

		glColor3dv( cube_col[ idx1 ].data() );
		glVertex3dv( cube_vrt[ idx1 ].data() );

		glColor3dv( cube_col[ idx2 ].data() );
		glVertex3dv( cube_vrt[ idx2 ].data() );
	}

	glEnd();

	for( int i = 0; i != 8; ++i )
	{
		renderText( cube_vrt[i].x, cube_vrt[i].y, cube_vrt[i].z, QString::number( i ) );
	}
}

void wgt_nbody_view::initializeGL()
{
	m_renderer = new QGLFramebufferObject( 1920, 1080 );
}

void wgt_nbody_view::paintGL( GLint x, GLint y, GLsizei width, GLsizei height, const nbvertex_t &camera_position, const nbvertex_t &up )
{
	glViewport( x, y, width, height );
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();

	GLfloat aspect = ((GLfloat)height)/((GLfloat)width);

	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();

	GLfloat	near = 1;
	GLfloat	far = 1000;

	gluPerspective( 60, 1.0/aspect, near, far );
	gluLookAt( camera_position.x, camera_position.y, camera_position.z,
			   m_mesh_sx*0.5, m_mesh_sy*0.5, m_mesh_sz*0.5,
			   up.x, up.y, up.z );

	glColor3f( 1,1,1 );
	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();

	renderText( 20 , 20, QString( "Step  = %1" ).arg( m_3body.get_step() ), QFont("Monospace") );
	renderText( 20 , 40, QString( "T     = %1" ).arg( m_3body.get_time() ), QFont("Monospace") );
	renderText( 20 , 60, QString( "Stars = %1" ).arg( m_3body.get_count() ), QFont("Monospace") );
	renderText( 20 , 80, QString( "dP    = %1 %" ).arg( m_3body.impulce_err(), 3, 'e', 2 ), QFont("Monospace") );
	renderText( 20 ,100, QString( "dL    = %1 %" ).arg( m_3body.impulce_moment_err(), 3, 'e', 2 ), QFont("Monospace") );
	renderText( 20 ,120, QString( "dE    = %1 %" ).arg( m_3body.energy_err(), 3, 'e', 2 ), QFont("Monospace") );

	paint_color_box();

	nbcoord_t intensity = 0.8;
	glColor3d( intensity*0.5, intensity*0.5, intensity );
	glDisable( GL_DEPTH_TEST );
	glLineWidth( 1 );
	glPointSize( 3 );
	glEnable( GL_POINT_SMOOTH );
	glEnable( GL_BLEND );
	glBlendFunc( GL_ONE, GL_ONE );

	glEnableClientState( GL_VERTEX_ARRAY );
	glVertexPointer( 3, GL_DOUBLE, 0, m_3body.get_vertites() );
	glDrawArrays( GL_POINTS, 0, (GLsizei)m_3body.get_count() );
	glDisableClientState( GL_VERTEX_ARRAY );
}

void wgt_nbody_view::render_file()
{
	makeCurrent();
	QString		out_dir( "/home/sas/Documents/prg/nbody/video" );
	if( !m_renderer->bind() )
	{
		qDebug() << "Can't bind QGLFramebufferObject";
		return;
	}
	paintGL( m_renderer->width(), m_renderer->height() );
	if( !m_renderer->release() )
	{
		qDebug() << "Can't release QGLFramebufferObject";
		return;
	}
	QImage	image( m_renderer->toImage() );
	if( image.isNull() )
	{
		qDebug() << "Can't convert QGLFramebufferObject to image";
		return;
	}

	{
		QPainter	p( &image );
		p.setRenderHints( QPainter::Antialiasing | QPainter::TextAntialiasing | QPainter::HighQualityAntialiasing );
		p.setPen( Qt::white );
		p.setFont( QFont("Monospace", 16) );
		p.drawLine( QPointF( 0, image.height()/2.0 ), QPointF( image.width(), image.height()/2.0 ) );
		p.drawLine( QPointF( image.width()/2.0, 0 ), QPointF( image.width()/2.0, image.height() ) );

		p.drawText( 20 , 20, QString( "Step  = %1" ).arg( m_3body.get_step() ) );
		p.drawText( 20 , 40, QString( "T     = %1" ).arg( m_3body.get_time() ) );
		p.drawText( 20 , 60, QString( "Stars = %1" ).arg( m_3body.get_count() ) );
		p.drawText( 20 , 80, QString( "dP    = %1 %" ).arg( m_3body.impulce_err(), 3, 'e', 2  ) );
		p.drawText( 20 ,100, QString( "dL    = %1 %" ).arg( m_3body.impulce_moment_err(), 3, 'e', 2 )  );
		p.drawText( 20 ,120, QString( "dE    = %1 %" ).arg( m_3body.energy_err(), 3, 'e', 2 )  );
	}
	QString name = out_dir + "/%1.png";

	if( !QDir( out_dir ).mkpath(".") )
	{
		qDebug() << "Can't mkpath" << out_dir;
		return;
	}

	name = name.arg( m_3body.get_step(), 8, 10,  QChar('0') );

	if( !image.save( name, "PNG" ) )
	{
		qDebug() << "Can't save image" << name;
		return;
	}
}

void wgt_nbody_view::paintGL( GLsizei width, GLsizei height )
{
	glViewport( 0, 0, width, height );
	qglClearColor( Qt::black );
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	nbvertex_t	center( m_mesh_sx*0.5, m_mesh_sy*0.5, m_mesh_sz*0.5 );
	nbcoord_t	dist = 200;
	int			sx = width/2;
	int			sy = height/2;

	paintGL( sx, sy, sx, sy, center - nbvertex_t( 0, dist, dist ), nbvertex_t( 0,0,1 ) );
	paintGL( 0, 0, sx, sy, center - nbvertex_t( 0, 0, dist ), nbvertex_t( 0,1,0 ) );
	paintGL( sx, 0, sx, sy, center + nbvertex_t( dist, 0, 0 ), nbvertex_t( 0,1,0 ) );
	paintGL( 0, sy, sx, sy, center - nbvertex_t( 0, dist, 0 ), nbvertex_t( 0,0,-1 ) );
}

void wgt_nbody_view::paintGL()
{
	paintGL( width(), height() );
}

void wgt_nbody_view::step()
{
	size_t	i = m_3body.get_step();

	size_t	w = 100;
	if( i % w == 0 )
	{
		m_3body.print_statistics();
		//render_file();
	}

	//nbcoord_t	step_time = omp_get_wtime();
	m_solver->step( 0.1/w );
	//qDebug() << "Step time" << step_time - omp_get_wtime();
}
