#include "nbody_data_stream.h"
#include "nbody_engine.h"
#include <QFile>
#include <QFileInfo>
#include <QDir>
#include <QTextStream>
#include <QDebug>

struct nbody_data_stream::data
{
	size_t			m_file_n;
	qint64			m_max_part_size;
	QTextStream		m_idx_stream;
	QFile			m_data;
	QFile			m_idx;
	QString			m_base_name;

	data() :
		m_file_n(0),
		m_max_part_size( 1024*1024*1024 )
	{
	}

	int open_data_file()
	{
		if( m_data.isOpen() )
		{
			m_data.close();
		}
		m_data.setFileName( make_dat_name( m_base_name, m_file_n ) );
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
		m_idx.setFileName( make_idx_name( m_base_name ) );
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

	if( !d->m_idx.isOpen() )
	{
		qDebug() << "Index not open yet!";
		return -1;
	}

	if( d->m_max_part_size > 0 && d->m_data.pos() >= d->m_max_part_size )
	{
		++d->m_file_n;
		if( 0 != d->open_data_file() )
		{
			qDebug() << "Can't d->open_data_file()";
			return -1;
		}
	}

	qint64		fpos( d->m_data.pos() );
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

	d->m_idx_stream << e->get_step() << get_idx_separator()
					<< e->get_time() << get_idx_separator()
					<< d->m_file_n << get_idx_separator()
					<< fpos << "\n";

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

QString nbody_data_stream::make_idx_name( const QString& file_base_name )
{
	return file_base_name + ".idx";
}

QString nbody_data_stream::make_dat_name( const QString& file_base_name, size_t file_n )
{
	return file_base_name + QString::number( file_n ) + ".dat";
}

QChar nbody_data_stream::get_idx_separator()
{
	return '\t';
}
