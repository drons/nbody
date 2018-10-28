#ifndef NBODY_DATA_STREAM_H
#define NBODY_DATA_STREAM_H

#include <qglobal.h>

class nbody_data;
class QString;
class QChar;

class nbody_data_stream
{
	struct	data;
	data*	d;

	nbody_data_stream(const nbody_data_stream&);
	nbody_data_stream& operator = (const nbody_data_stream&);
public:
	nbody_data_stream();
	virtual ~nbody_data_stream();
	virtual int write(const nbody_data* bdata);
	int open(const QString& file_base_name, qint64 max_part_size);
	void close();

	static QString make_idx_name(const QString& file_base_name);
	static QString make_dat_name(const QString& file_base_name, size_t part_n);
	static QChar get_idx_separator();
};

#endif // NBODY_DATA_STREAM_H
