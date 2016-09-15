#include "nbody_data_stream_reader.h"
#include "nbody_data_stream.h"
#include "nbody_engine.h"
#include <vector>
#include <map>
#include <limits>
#include <QFile>
#include <QTextStream>
#include <QDebug>

struct nbody_data_stream_reader::data
{
	struct item
	{
		size_t		frame;
		size_t		step;
		nbcoord_t	time;
		size_t		file_n;
		qint64		file_pos;
	};

	std::vector<item>			m_frames;
	std::map<size_t,size_t>		m_step2frame;
	std::map<nbcoord_t,size_t>	m_time2frame;
	QString						m_file_base_name;
	QFile						m_file;
	size_t						m_file_n;
	size_t						m_current_frame;

	data() :
		m_file_n( std::numeric_limits<size_t>::max() ),
		m_current_frame( std::numeric_limits<size_t>::max() )
	{
	}
};

nbody_data_stream_reader::nbody_data_stream_reader() : d( new data() )
{

}

nbody_data_stream_reader::~nbody_data_stream_reader()
{
	delete d;
}

int nbody_data_stream_reader::load( const QString& file_base_name )
{
	close();

	QFile	idx( nbody_data_stream::make_idx_name( file_base_name ) );

	if( !idx.open( QFile::ReadOnly ) )
	{
		qDebug() << "Can't open index file" << idx.fileName() << idx.errorString();
		return -1;
	}

	QTextStream	stream( &idx );

	while( !stream.atEnd() )
	{
		QString			line( stream.readLine() );
		QStringList		parts( line.split( nbody_data_stream::get_idx_separator() ) );

		if( parts.size() != 4 )
		{
			qDebug() << "Incomplete data line " << line;
			return -1;
		}

		data::item	i;
		bool		ok[4] = { false, false, false, false };

		i.frame = d->m_frames.size();
		i.step = (size_t)parts[0].toULongLong( &ok[0] );
		i.time = (nbcoord_t)parts[1].toDouble( &ok[1] );
		i.file_n = (size_t)parts[2].toULongLong( &ok[2] );
		i.file_pos = (qint64)parts[3].toLongLong( &ok[3] );

		if( !( ok[0] && ok[1] && ok[2] && ok[3] ) )
		{
			qDebug() << "Invalid data line" << line;
			return -1;
		}

		d->m_frames.push_back( i );
		d->m_step2frame[i.step] = i.frame;
		d->m_time2frame[i.time] = i.frame;
	}

	d->m_file_base_name = file_base_name;

	return seek( 0 );
}

void nbody_data_stream_reader::close()
{
	d->m_frames.clear();
	d->m_step2frame.clear();
	d->m_time2frame.clear();
	d->m_file_base_name.clear();
	d->m_file.close();
	d->m_file_n = std::numeric_limits<size_t>::max();
	d->m_current_frame = std::numeric_limits<size_t>::max();
}

size_t nbody_data_stream_reader::get_frame_count()
{
	return d->m_frames.size();
}

size_t nbody_data_stream_reader::get_steps_count()
{
	if( d->m_frames.size() == 0 )
	{
		return 0;
	}
	return d->m_frames.back().step;
}

nbcoord_t nbody_data_stream_reader::get_max_time()
{
	if( d->m_frames.size() == 0 )
	{
		return 0;
	}
	return d->m_frames.back().time;
}

int nbody_data_stream_reader::seek( size_t frame )
{
	if( frame == d->m_current_frame )
	{
		return 0;
	}

	if( frame >= d->m_frames.size() )
	{
		qDebug() << "Can't seek to frame" << frame << "Out of range";
		return -1;
	}

	const data::item&	i( d->m_frames[frame] );

	if( d->m_file_n != i.file_n ||
		!d->m_file.isOpen() )
	{
		d->m_file.close();
		d->m_file.setFileName( nbody_data_stream::make_dat_name( d->m_file_base_name, i.file_n ) );
		if( !d->m_file.open( QFile::ReadOnly ) )
		{
			qDebug() << "Can't open file" << d->m_file.fileName() << d->m_file.errorString();
			return -1;
		}
	}

	if( !d->m_file.seek( i.file_pos ) )
	{
		qDebug() << "Can't seek file" << d->m_file.fileName()
				 << "To offset" << i.file_pos << d->m_file.errorString();
		return -1;
	}

	d->m_current_frame = frame;

	return 0;
}

int nbody_data_stream_reader::read( nbody_engine* e )
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

	seek( d->m_current_frame );

	qint64		fpos( d->m_file.pos() );
	QByteArray	ybuf( d->m_file.read( e->y()->size() ) );

	if( ybuf.size() != (int)e->y()->size() )
	{
		qDebug() << "Can't read file" << d->m_file.fileName()
				 << "Need to read" << e->y()->size() << "But only" << ybuf.size() << "from pos" << fpos
				 << d->m_file.errorString();
		return -1;
	}

	e->write_buffer( e->y(), ybuf.data() );

	if( d->m_current_frame < d->m_frames.size() - 1 )
	{
		seek( d->m_current_frame + 1 );
	}

	return 0;
}
