#include "nbody_data_stream.h"
#include "nbody_data_stream_reader.h"
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
	bool			m_header_written;

	data() :
		m_file_n(0),
		m_max_part_size(1024 * 1024 * 1024),
		m_header_written(false)
	{
	}

	int open_data_file(bool append = false)
	{
		if(m_data.isOpen())
		{
			m_data.close();
		}
		m_data.setFileName(make_dat_name(m_base_name, m_file_n));
		QFile::OpenMode openmode = append ? QFile::Append : QFile::WriteOnly;
		if(!m_data.open(openmode))
		{
			qDebug() << "Can't open file" << m_data.fileName() << m_data.errorString();
			return -1;
		}
		return 0;
	}

	int open_index_file(bool append)
	{
		if(m_idx.isOpen())
		{
			m_idx.close();
		}
		m_idx.setFileName(make_idx_name(m_base_name));
		QFile::OpenMode openmode = append ? QFile::Append : QFile::WriteOnly;
		if(!m_idx.open(openmode))
		{
			qDebug() << "Can't open file" << m_idx.fileName() << m_idx.errorString();
			return -1;
		}
		m_idx_stream.setDevice(&m_idx);
		return 0;
	}

	int write_header(const nbody_data* bdata)
	{
		m_idx_stream << "#coord_size " << sizeof(nbcoord_t) << "\n";
		m_idx_stream << "#body_count " << bdata->get_count() << "\n";
		m_idx_stream << "#box_size " << bdata->get_box_size() << "\n";
		{
			QFile	col(make_col_name(m_base_name));
			if(!col.open(QFile::WriteOnly))
			{
				qDebug() << "Can't open file" << col.fileName() << col.errorString();
				return -1;
			}
			qint64		sz(sizeof(nbcolor_t)*bdata->get_count());
			if(sz != col.write(reinterpret_cast<const char*>(bdata->get_color()), sz))
			{
				qDebug() << "Can't write file" << col.fileName() << col.errorString();
				return -1;
			}
		}
		{
			QFile	mass(make_mass_name(m_base_name));
			if(!mass.open(QFile::WriteOnly))
			{
				qDebug() << "Can't open file" << mass.fileName() << mass.errorString();
				return -1;
			}
			qint64		sz(sizeof(nbcoord_t)*bdata->get_count());
			if(sz != mass.write(reinterpret_cast<const char*>(bdata->get_mass()), sz))
			{
				qDebug() << "Can't write file" << mass.fileName() << mass.errorString();
				return -1;
			}
		}
		m_header_written = true;
		return 0;
	}
};

nbody_data_stream::nbody_data_stream() : d(new data())
{

}

nbody_data_stream::~nbody_data_stream()
{
	delete d;
}

int nbody_data_stream::write(const nbody_data* bdata)
{
	if(bdata == NULL)
	{
		qDebug() << "data == NULL";
		return -1;
	}

	if(!d->m_idx.isOpen())
	{
		qDebug() << "Index not open yet!";
		return -1;
	}

	if((!d->m_header_written) && 0 != d->write_header(bdata))
	{
		qDebug() << "Write header failed";
		return -1;
	}

	if(d->m_max_part_size > 0 && d->m_data.pos() >= d->m_max_part_size)
	{
		++d->m_file_n;
		if(0 != d->open_data_file())
		{
			qDebug() << "Can't d->open_data_file()";
			return -1;
		}
	}

	qint64		fpos(d->m_data.pos());
	qint64		sz(sizeof(nbvertex_t)*bdata->get_count());
	if(sz != d->m_data.write(reinterpret_cast<const char*>(bdata->get_vertites()), sz))
	{
		qDebug() << "Can't write file" << d->m_data.fileName()
				 << d->m_data.errorString();
		return -1;
	}
	if(sz != d->m_data.write(reinterpret_cast<const char*>(bdata->get_velosites()), sz))
	{
		qDebug() << "Can't write file" << d->m_data.fileName()
				 << d->m_data.errorString();
		return -1;
	}

	d->m_data.flush();

	d->m_idx_stream << bdata->get_step() << get_idx_separator()
					<< bdata->get_time() << get_idx_separator()
					<< d->m_file_n << get_idx_separator()
					<< fpos << "\n";

	d->m_idx_stream.flush();

	return 0;
}

int nbody_data_stream::open(const QString& name, qint64 max_part_size,
							const nbody_data_stream_reader* append_to)
{
	QFileInfo	finfo(name);
	finfo.dir().mkpath(".");

	d->m_base_name = name;
	d->m_max_part_size = max_part_size;
	bool append_mode = append_to != NULL;
	if(append_mode)
	{
		d->m_file_n = append_to->get_last_file_n();
		d->m_header_written = true;
	}
	else
	{
		d->m_file_n = 0;
	}

	if(0 != d->open_data_file(append_mode))
	{
		qDebug() << "Can't d->open_data_file()";
		return -1;
	}

	if(0 != d->open_index_file(append_mode))
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

QString nbody_data_stream::make_idx_name(const QString& file_base_name)
{
	return file_base_name + ".idx";
}

QString nbody_data_stream::make_dat_name(const QString& file_base_name, size_t file_n)
{
	return file_base_name + QString::number(file_n) + ".dat";
}

QString nbody_data_stream::make_col_name(const QString& file_base_name)
{
	return file_base_name + ".col";
}

QString nbody_data_stream::make_mass_name(const QString& file_base_name)
{
	return file_base_name + ".mass";
}

QChar nbody_data_stream::get_idx_separator()
{
	return '\t';
}
