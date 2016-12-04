#ifndef NBODY_FRAME_COMPRESSOR_H
#define NBODY_FRAME_COMPRESSOR_H

#include <QObject>
#include <QImage>

class nbody_frame_compressor
{
	QString		m_destination;
	QString		m_name_tmpl;
public:
	nbody_frame_compressor();
	bool set_destination( const QString& );
	void push_frame( const QImage& f, size_t frame_n );
};

#endif // NBODY_FRAME_COMPRESSOR_H
