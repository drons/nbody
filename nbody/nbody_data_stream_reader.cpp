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
	std::map<size_t, size_t>		m_step2frame;
	std::map<nbcoord_t, size_t>	m_time2frame;
	QString						m_file_base_name;
	QFile						m_file;
	size_t						m_file_n;
	size_t						m_current_frame;
	size_t						m_coord_size;
	size_t						m_body_count;
	size_t						m_box_size;
	data() :
		m_file_n(std::numeric_limits<size_t>::max()),
		m_current_frame(std::numeric_limits<size_t>::max()),
		m_coord_size(0),
		m_body_count(0),
		m_box_size(0)
	{
	}

	void parse_header_line(const QString& line)
	{
		QStringList	list(line.split(" "));
		if(list.size() != 2)
		{
			qDebug() << "Invalid header line" << line;
			return;
		}
		bool	ok = false;
		size_t	value = static_cast<size_t>(list[1].toULongLong(&ok));
		if(!ok)
		{
			qDebug() << "Invalid header line" << line;
			return;
		}
		if(list[0] == "#coord_size")
		{
			m_coord_size = value;
		}
		else if(list[0] == "#body_count")
		{
			m_body_count = value;
		}
		else if(list[0] == "#box_size")
		{
			m_box_size = value;
		}
	}
};

nbody_data_stream_reader::nbody_data_stream_reader() : d(new data())
{

}

nbody_data_stream_reader::~nbody_data_stream_reader()
{
	delete d;
}

int nbody_data_stream_reader::load(const QString& file_base_name)
{
	close();

	QFile	idx(nbody_data_stream::make_idx_name(file_base_name));

	if(!idx.open(QFile::ReadOnly))
	{
		qDebug() << "Can't open index file" << idx.fileName() << idx.errorString();
		return -1;
	}

	QTextStream	stream(&idx);

	while(!stream.atEnd())
	{
		QString			line(stream.readLine());
		QStringList		parts(line.split(nbody_data_stream::get_idx_separator()));

		if(line.startsWith("#"))
		{
			d->parse_header_line(line);
			continue;
		}
		if(parts.size() != 4)
		{
			qDebug() << "Incomplete data line " << line;
			return -1;
		}

		data::item	i;
		bool		ok[4] = { false, false, false, false };

		i.frame = d->m_frames.size();
		i.step = static_cast<size_t>(parts[0].toULongLong(&ok[0]));
		i.time = static_cast<nbcoord_t>(parts[1].toDouble(&ok[1]));
		i.file_n = static_cast<size_t>(parts[2].toULongLong(&ok[2]));
		i.file_pos = static_cast<qint64>(parts[3].toLongLong(&ok[3]));

		if(!(ok[0] && ok[1] && ok[2] && ok[3]))
		{
			qDebug() << "Invalid data line" << line;
			return -1;
		}

		d->m_frames.push_back(i);
		d->m_step2frame[i.step] = i.frame;
		d->m_time2frame[i.time] = i.frame;
	}

	if(d->m_body_count == 0 || d->m_coord_size == 0 || d->m_box_size == 0)
	{
		qDebug() << "Invalid file header "
				 << "'coord_size' == " << d->m_coord_size
				 << "'body_count' == " << d->m_body_count
				 << "'box_size' == " << d->m_box_size;
		return -1;
	}
	if(d->m_coord_size != sizeof(nbcoord_t))
	{
		qDebug() << "Invalid file header "
				 << "'coord_size' == " << d->m_coord_size
				 << "must be " << sizeof(nbcoord_t);
		return -1;
	}
	d->m_file_base_name = file_base_name;

	return seek(0);
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

size_t nbody_data_stream_reader::get_frame_count() const
{
	return d->m_frames.size();
}

size_t nbody_data_stream_reader::get_steps_count() const
{
	if(d->m_frames.size() == 0)
	{
		return 0;
	}
	return d->m_frames.back().step;
}

nbcoord_t nbody_data_stream_reader::get_max_time() const
{
	if(d->m_frames.size() == 0)
	{
		return 0;
	}
	return d->m_frames.back().time;
}

int nbody_data_stream_reader::seek(size_t frame)
{
	if(frame == d->m_current_frame)
	{
		return 0;
	}

	if(frame >= d->m_frames.size())
	{
		qDebug() << "Can't seek to frame" << frame << "Out of range";
		return -1;
	}

	const data::item&	i(d->m_frames[frame]);

	if(d->m_file_n != i.file_n || !d->m_file.isOpen())
	{
		d->m_file.close();
		d->m_file.setFileName(nbody_data_stream::make_dat_name(d->m_file_base_name, i.file_n));
		if(!d->m_file.open(QFile::ReadOnly))
		{
			qDebug() << "Can't open file" << d->m_file.fileName() << d->m_file.errorString();
			return -1;
		}
	}

	if(!d->m_file.seek(i.file_pos))
	{
		qDebug() << "Can't seek file" << d->m_file.fileName()
				 << "To offset" << i.file_pos << d->m_file.errorString();
		d->m_file.close();
		return -1;
	}

	d->m_current_frame = frame;

	return 0;
}

size_t nbody_data_stream_reader::get_current_frame() const
{
	return d->m_current_frame;
}

size_t nbody_data_stream_reader::get_body_count() const
{
	return d->m_body_count;
}

size_t nbody_data_stream_reader::get_coord_size() const
{
	return d->m_coord_size;
}

nbcoord_t nbody_data_stream_reader::get_box_size() const
{
	return d->m_box_size;
}

size_t nbody_data_stream_reader::get_last_file_n() const
{
	if(d->m_frames.empty())
	{
		return 0;
	}
	return d->m_frames.back().file_n;
}

int nbody_data_stream_reader::read(nbody_data* bdata)
{
	if(bdata == NULL)
	{
		qDebug() << "bdata == NULL";
		return -1;
	}
	if(bdata->get_count() != d->m_body_count)
	{
		qDebug() << "Invalid body count in destination buffer" << bdata->get_count() << "must be" << d->m_body_count;
		return -1;
	}

	if(0 != seek(d->m_current_frame))
	{
		qDebug() << "Can't seek to current frame" << d->m_current_frame;
		return -1;
	}

	qint64				fpos(d->m_file.pos());
	qint64				sz(sizeof(nbvertex_t)*bdata->get_count());
	const data::item&	frame(d->m_frames[ d->m_current_frame ]);

	if(sz != d->m_file.read(reinterpret_cast<char*>(bdata->get_vertites()), sz))
	{
		qDebug() << "Can't read file" << d->m_file.fileName()
				 << d->m_file.errorString() << "from pos" << fpos;
		return -1;
	}
	if(sz != d->m_file.read(reinterpret_cast<char*>(bdata->get_velosites()), sz))
	{
		qDebug() << "Can't read file" << d->m_file.fileName()
				 << d->m_file.errorString() << "from pos" << fpos + sz;
		return -1;
	}

	bdata->set_time(frame.time);
	bdata->set_step(frame.step);
	if(d->m_current_frame < d->m_frames.size() - 1)
	{
		seek(d->m_current_frame + 1);
	}
	{
		QFile	col(nbody_data_stream::make_col_name(d->m_file_base_name));
		if(!col.open(QFile::ReadOnly))
		{
			qDebug() << "Can't open file" << col.fileName() << col.errorString();
			return -1;
		}
		qint64		col_sz(sizeof(nbcolor_t)*bdata->get_count());
		if(col_sz != col.read(reinterpret_cast<char*>(bdata->get_color()), col_sz))
		{
			qDebug() << "Can't read file" << col.fileName() << col.errorString();
			return -1;
		}
	}
	{
		QFile	mass(nbody_data_stream::make_mass_name(d->m_file_base_name));
		if(!mass.open(QFile::ReadOnly))
		{
			qDebug() << "Can't open file" << mass.fileName() << mass.errorString();
			return -1;
		}
		qint64		mass_sz(sizeof(nbcoord_t)*bdata->get_count());
		if(mass_sz != mass.read(reinterpret_cast<char*>(bdata->get_mass()), mass_sz))
		{
			qDebug() << "Can't read file" << mass.fileName() << mass.errorString();
			return -1;
		}
	}

	return 0;
}
