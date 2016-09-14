#include "nbody_data_stream.h"
#include "nbody_engine.h"
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QTextStream>
#include <QDebug>

struct nbody_data_stream::data
{
	quint64			m_file_n;
	qint64			m_max_part_size;
	QTextStream		m_idx_stream;
	QFile			m_data;
	QFile			m_idx;
	QString			m_base_name;
	QChar			m_separator;

	data() :
		m_file_n(0),
		m_max_part_size( 1024*1024*1024 ),
		m_separator( '\t' )
	{
	}

	int open_data_file()
	{
		if( m_data.isOpen() )
		{
			m_data.close();
		}
		m_data.setFileName( m_base_name + QString::number( m_file_n ) + ".dat" );
		if( !m_data.open( QFile::WriteOnly ) )
		{
			qDebug() << "Can't open file" << m_data.fileName() << m_data.errorString();
			return -1;
		}
		return 0;
	}

	int open_index_file()
	{
		if( m_idx.isOpen() )
		{
			m_idx.close();
		}
		m_idx.setFileName( m_base_name + ".idx" );
		if( !m_idx.open( QFile::WriteOnly ) )
		{
			qDebug() << "Can't open file" << m_idx.fileName() << m_idx.errorString();
			return -1;
		}
		m_idx_stream.setDevice( &m_idx );
		return 0;
	}
};

nbody_data_stream::nbody_data_stream() : d( new data() )
{

}

nbody_data_stream::~nbody_data_stream()
{
	delete d;
}

int nbody_data_stream::write( nbody_engine* e )
{
	if( e == NULL )
	{
		qDebug() << "e == NULL";
		return -1;
	}

	if( e->y() == NULL )
	{
		qDebug() << "e->y() == NULL";
		return -1;
	}

	qint64	fpos( d->m_data.pos() );

	if( d->m_max_part_size > 0 && fpos >= d->m_max_part_size )
	{
		++d->m_file_n;
		if( 0 != d->open_data_file() )
		{
			qDebug() << "Can't d->open_data_file()";
			return -1;
		}
	}

	QByteArray	ybuf;
	ybuf.resize( e->y()->size() );
	e->read_buffer( ybuf.data(), e->y() );

	if( ybuf.size() != d->m_data.write( ybuf ) )
	{
		qDebug() << "Can't write file" << d->m_data.fileName()
				 << d->m_data.errorString();
		return -1;
	}

	d->m_data.flush();

	d->m_idx_stream << e->get_step() << d->m_separator
					<< e->get_time() << d->m_separator
					<< d->m_file_n << d->m_separator
					<< d->m_data.pos() << "\n";

	d->m_idx_stream.flush();

	return 0;
}

int nbody_data_stream::open( const QString& name, qint64 max_part_size )
{
	QFileInfo	finfo( name );
	finfo.dir().mkpath(".");

	d->m_base_name = name;
	d->m_file_n = 0;
	d->m_max_part_size = max_part_size;

	if( 0 != d->open_data_file() )
	{
		qDebug() << "Can't d->open_data_file()";
		return -1;
	}

	if( 0 != d->open_index_file() )
	{
		qDebug() << "Can't d->open_index_file()";
		return -1;
	}

	return 0;
}

void nbody_data_stream::close()
{
	d->m_base_name.clear();
	d->m_file_n = 0;
	d->m_idx.close();
	d->m_data.close();
}
