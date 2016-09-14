#ifndef NBODY_DATA_STREAM_H
#define NBODY_DATA_STREAM_H

#include <qglobal.h>

class nbody_engine;
class QString;

class nbody_data_stream
{
	struct	data;
	data*	d;
public:
	nbody_data_stream();
	virtual ~nbody_data_stream();
	virtual int write( nbody_engine* );
	int open( const QString& file_base_name, qint64 max_part_size );
	void close();
};

#endif // NBODY_DATA_STREAM_H
