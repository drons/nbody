#include "nbody_frame_compressor.h"

#include <QDir>
#include <QDebug>

nbody_frame_compressor::nbody_frame_compressor()
{

}

bool nbody_frame_compressor::set_destination( const QString& d )
{
	if( !QDir( d ).mkpath(".") )
	{
		qDebug() << "Can't mkpath" << d;
		return false;
	}

	m_destination = d;

	m_name_tmpl = d + "/%1.png";

	return true;
}

void nbody_frame_compressor::push_frame( const QImage& frame, size_t frame_n )
{
	QString		frame_name( m_name_tmpl.arg( frame_n, 8, 10,  QChar('0') ) );

	if( !frame.save( frame_name, "PNG" ) )
	{
		qDebug() << "Can't save image" << frame_name;
		return;
	}
}
