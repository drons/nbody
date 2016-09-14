#include <QString>
#include <QFile>
#include <QtTest>

#include "nbody_solvers.h"
#include "nbody_engines.h"
#include "nbody_data_stream.h"

class test_nbody_stream : public QObject
{
	Q_OBJECT

	nbody_data			data;
	nbody_engine*		e;
	nbody_solver*		s;
	nbody_data_stream*	stream;
public:
	test_nbody_stream();
	~test_nbody_stream();
private Q_SLOTS:
	void initTestCase();
	void cleanupTestCase();
	void run();
};

test_nbody_stream::test_nbody_stream() :
	e( new nbody_engine_simple() ),
    s( new nbody_solver_euler() ),
    stream( new nbody_data_stream() )
{

}

test_nbody_stream::~test_nbody_stream()
{
	delete s;
	delete e;
	delete stream;
}

void test_nbody_stream::initTestCase()
{
	nbcoord_t			box_size = 100;

	qDebug() << "Solver =" << s->type_name() << "engine" << e->type_name();
	data.make_universe( box_size, box_size, box_size );

	e->init( &data );
	s->set_time_step( 1e-3, 3e-2 );
	s->set_engine( e );
}

void test_nbody_stream::cleanupTestCase()
{
}

void test_nbody_stream::run()
{
	s->set_time_step( 0.1, 0.1 );
	//open stream with 1 byte limit
	QTEST_ASSERT( 0 == stream->open( "/tmp/stream-test/new", 1 ) );
	QTEST_ASSERT( 0 == s->run( &data, stream, 0.31, 0.1, 0.1 ) );

	stream->close();

	QTEST_ASSERT( QFile::exists( "/tmp/stream-test/new.idx" ) );
	QTEST_ASSERT( QFile::exists( "/tmp/stream-test/new0.dat" ) );
	QTEST_ASSERT( QFile::exists( "/tmp/stream-test/new1.dat" ) );
	QTEST_ASSERT( QFile::exists( "/tmp/stream-test/new2.dat" ) );

	QFile	idx("/tmp/stream-test/new.idx");

	QTEST_ASSERT( idx.open( QFile::ReadOnly ) );

	QTEST_ASSERT( QString(idx.readAll()).split( QChar('\n'), QString::SkipEmptyParts ).size() == 3 );
}

int main(int argc, char *argv[])
{
	test_nbody_stream tc1;
	return QTest::qExec( &tc1, argc, argv );
}

#include "test_nbody_stream.moc"
