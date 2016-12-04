#ifndef NBODY_FRAME_COMPRESSOR_H
#define NBODY_FRAME_COMPRESSOR_H

#include <QObject>
#include <QImage>
#include <QList>
#include <QtConcurrentRun>

class nbody_frame_compressor
{
	QString					m_destination;
	QString					m_name_tmpl;
	QList< QFuture<bool> >	m_results;
public:
	nbody_frame_compressor();
	~nbody_frame_compressor();
	bool set_destination( const QString& );
	void push_frame( const QImage& f, size_t frame_n );
private:
	void wait_results( int max_queue_size );
};

#endif // NBODY_FRAME_COMPRESSOR_H
