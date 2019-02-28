#ifndef NBODY_DATA_STREAM_H
#define NBODY_DATA_STREAM_H

#include <qglobal.h>
#include "nbody_export.h"

class nbody_data;
class nbody_data_stream_reader;
class QString;
class QChar;

class NBODY_DLL nbody_data_stream
{
	struct	data;
	data*	d;

	nbody_data_stream(const nbody_data_stream&);
	nbody_data_stream& operator = (const nbody_data_stream&);
public:
	nbody_data_stream();
	virtual ~nbody_data_stream();
	virtual int write(const nbody_data* bdata);
	int open(const QString& file_base_name, qint64 max_part_size,
			 const nbody_data_stream_reader* append_to = NULL);
	void close();

	static QString make_idx_name(const QString& file_base_name);
	static QString make_dat_name(const QString& file_base_name, size_t part_n);
	static QString make_col_name(const QString& file_base_name);
	static QString make_mass_name(const QString& file_base_name);
	static QChar get_idx_separator();
};

#endif // NBODY_DATA_STREAM_H
