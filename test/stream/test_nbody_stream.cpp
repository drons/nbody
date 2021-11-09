#include <QString>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QtTest>

#include "nbody_solvers.h"
#include "nbody_engines.h"
#include "nbody_data_stream.h"
#include "nbody_data_stream_reader.h"

class test_nbody_stream : public QObject
{
	Q_OBJECT

	nbody_data			m_data;
	nbody_engine*		m_e;
	nbody_solver*		m_s;
	nbody_data_stream*	m_stream;
	QString				m_tmp;
public:
	test_nbody_stream();
	~test_nbody_stream();
private Q_SLOTS:
	void initTestCase();
	void cleanupTestCase();
	void run();
	void negative_branch();
};

test_nbody_stream::test_nbody_stream() :
	m_e(new nbody_engine_simple()),
	m_s(new nbody_solver_euler()),
	m_stream(new nbody_data_stream()),
	m_tmp(QDir::tempPath())
{
}

test_nbody_stream::~test_nbody_stream()
{
	delete m_s;
	delete m_e;
	delete m_stream;
}

void test_nbody_stream::initTestCase()
{
	nbcoord_t			box_size = 100;

	qDebug() << "Solver =" << m_s->type_name() << "engine" << m_e->type_name();
	m_data.make_universe(128, box_size, box_size, box_size);

	QVERIFY(m_e->init(&m_data));
	m_s->set_time_step(1e-3, 3e-2);
	m_s->set_engine(m_e);
}

void test_nbody_stream::cleanupTestCase()
{
}

void test_nbody_stream::run()
{
	m_s->set_time_step(0.01, 0.1);
	//open stream with 2 frames per data file limit
	QVERIFY(0 == m_stream->open(m_tmp + "/stream-test/new", 14000));
	QVERIFY(0 == m_s->run(&m_data, m_stream, 0.31, 0.1, 0.1));

	m_stream->close();

	QVERIFY(QFile::exists(m_tmp + "/stream-test/new.idx"));
	QVERIFY(QFile::exists(m_tmp + "/stream-test/new0.dat"));
	QVERIFY(QFile::exists(m_tmp + "/stream-test/new1.dat"));
	QVERIFY(QFile::exists(m_tmp + "/stream-test/new2.dat"));

	QFile	idx(m_tmp + "/stream-test/new.idx");

	QVERIFY(idx.open(QFile::ReadOnly));

	QVERIFY(QString(idx.readAll()).split(QChar('\n'), QString::SkipEmptyParts).size() == 8);

	{
		nbody_data_stream_reader	reader;
		nbody_data					data, expected;

		expected.resize(m_data.get_count());
		data.resize(m_data.get_count());

		QVERIFY(0 == reader.load(m_tmp + "/stream-test/new"));
		QVERIFY(5 == reader.get_frame_count());
		QVERIFY(4 == reader.get_steps_count());
		QVERIFY(0.4 == reader.get_max_time());
		QVERIFY(m_data.get_count() == reader.get_body_count());
		QVERIFY(sizeof(nbcoord_t) == reader.get_coord_size());

		expected = m_data;
		m_e->get_data(&expected);

		QVERIFY(0 == reader.seek(0));
		QVERIFY(0 != reader.seek(77));

		QVERIFY(0 == reader.seek(0));
		QVERIFY(0 == reader.read(&data));
		QVERIFY(!data.is_equal(expected));

		QVERIFY(0 == reader.seek(reader.get_frame_count() - 1));
		QVERIFY(0 == reader.read(&data));
		QVERIFY(data.is_equal(expected));

		QVERIFY(0 == reader.seek(0));

		QVERIFY(0 == reader.read(&data));
		QVERIFY(!data.is_equal(expected));
		QVERIFY(0 == data.get_time());
		QVERIFY(0 == data.get_step());

		QVERIFY(0 == reader.read(&data));
		QVERIFY(!data.is_equal(expected));

		QVERIFY(0 == reader.read(&data));
		QVERIFY(!data.is_equal(expected));

		QVERIFY(0 == reader.read(&data));
		QVERIFY(!data.is_equal(expected));
		QVERIFY(0.3 == data.get_time());
		QVERIFY(3 == data.get_step());

		QVERIFY(0 == reader.read(&data));
		QVERIFY(data.is_equal(expected));
		QVERIFY(0.4 == data.get_time());
		QVERIFY(4 == data.get_step());
	}
}

void test_nbody_stream::negative_branch()
{
	{
		nbody_data_stream	stream;
		QVERIFY(0 != stream.write(NULL));
	}
	{
		nbody_data_stream	stream;
		nbody_data			data;
		data.make_universe(32, 1, 1, 1);
		QVERIFY(0 != stream.open("/dev/null/:nameless", 1000));
		QVERIFY(0 != stream.write(&data));
	}
	{
		nbody_data_stream	stream;
		nbody_data			data;
		data.make_universe(32, 1, 1, 1);
		QVERIFY(0 != stream.write(&data));
	}

	{
		nbody_data_stream	stream;
		nbody_data			data;
		data.make_universe(32, 1, 1, 1);
		QVERIFY(0 == stream.open(m_tmp + "/nbody_test/00/stream", 1000));
		QVERIFY(0 == stream.open(m_tmp + "/nbody_test/01/stream", 1000));
		QVERIFY(0 == stream.write(&data));
	}

	{
		nbody_data_stream	stream;
		nbody_data			data;
		data.make_universe(32, 1, 1, 1);
		QDir(m_tmp + "/nbody_test/0/stream0.dat").mkpath(".");
		QVERIFY(0 != stream.open(m_tmp + "/nbody_test/0/stream", 1));
		QVERIFY(0 != stream.write(&data));
		QDir("/tmp").rmpath("nbody_test/0");
	}
	{
		nbody_data_stream	stream;
		nbody_data			data;
		data.make_universe(32, 1, 1, 1);
		QDir(m_tmp + "/nbody_test/1/stream.idx").mkpath(".");
		QVERIFY(0 != stream.open(m_tmp + "/nbody_test/1/stream", 1));
		QVERIFY(0 != stream.write(&data));
		QDir("/tmp").rmpath("nbody_test/1");
	}
	{
		nbody_data_stream	stream;
		nbody_data			data;
		data.make_universe(32, 1, 1, 1);
		QDir(m_tmp + "/nbody_test/2/stream1.dat").mkpath(".");
		QVERIFY(0 == stream.open(m_tmp + "/nbody_test/2/stream", 1));
		QVERIFY(0 == stream.write(&data));
		QVERIFY(0 != stream.write(&data));
		QDir("/tmp").rmpath("nbody_test/2");
	}

	{
		nbody_data_stream_reader	stream;
		QVERIFY(0 != stream.load("/dev/null/:nameless"));
	}

	{
		nbody_data_stream_reader	stream;
		QDir(m_tmp + "/nbody_test/3").mkpath(".");
		{
			QFile	f(m_tmp + "/nbody_test/3/stream.idx");
			f.open(QFile::WriteOnly);
			f.write(QByteArray("0\t1\t2\n"));
		}
		QVERIFY(0 != stream.load(m_tmp + "/nbody_test/3/stream"));
	}

	{
		nbody_data_stream_reader	stream;
		QDir(m_tmp + "/nbody_test/4").mkpath(".");
		{
			QFile	f(m_tmp + "/nbody_test/4/stream.idx");
			f.open(QFile::WriteOnly);
			f.write(QByteArray("a\tb\tc\td\n"));
		}
		QVERIFY(0 != stream.load(m_tmp + "/nbody_test/4/stream"));
	}

	{
		nbody_data_stream_reader	stream;
		QVERIFY(0 == stream.get_frame_count());
		QVERIFY(0 == stream.get_steps_count());
		QVERIFY(0 == stream.get_max_time());
	}

	{
		{
			nbody_data_stream	stream;
			QVERIFY(0 == stream.open(m_tmp + "/nbody_test/5/stream", 1));
			QVERIFY(0 == stream.write(&m_data));
			QVERIFY(0 == stream.write(&m_data));
			QFile::remove(m_tmp + "/nbody_test/5/stream0.dat");
		}

		nbody_data_stream_reader	stream;
		QVERIFY(0 != stream.load(m_tmp + "/nbody_test/5/stream"));
	}

	{
		{
			nbody_data_stream	stream;
			QVERIFY(0 == stream.open(m_tmp + "/nbody_test/6/stream", 1));
			QVERIFY(0 == stream.write(&m_data));
			QVERIFY(0 == stream.write(&m_data));
			QFile::remove(m_tmp + "/nbody_test/6/stream1.dat");
		}

		nbody_data_stream_reader	stream;
		nbody_data					data;
		data.resize(m_data.get_count());
		QVERIFY(0 == stream.load(m_tmp + "/nbody_test/6/stream"));
		QVERIFY(0 == stream.read(&data));
		QVERIFY(0 != stream.read(&data));
	}

	{
		{
			nbody_data_stream	stream;
			QVERIFY(0 == stream.open(m_tmp + "/nbody_test/7/stream", 1));
			QVERIFY(0 == stream.write(&m_data));
			QVERIFY(0 == stream.write(&m_data));
			QVERIFY(0 == stream.write(&m_data));
		}
		{
			QFile	f(m_tmp + "/nbody_test/7/stream.idx");
			f.open(QFile::WriteOnly);
			f.write(QByteArray("#coord_size " +
							   QByteArray::number(static_cast<int>(sizeof(nbcoord_t))) +
							   "\n"));
			f.write(QByteArray("#body_count 256\n"));
			f.write(QByteArray("#box_size 100\n"));
			f.write(QByteArray("1\t1\t0\t0\n"));
			f.write(QByteArray("1\t1\t1\t-1\n"));  //Negative file offset
			f.write(QByteArray("1\t1\t2\t-1\n"));  //Negative file offset
		}
		{
			nbody_data_stream_reader	stream;
			nbody_data					data;
			data.resize(m_data.get_count());
			QVERIFY(0 == stream.load(m_tmp + "/nbody_test/7/stream"));
			QVERIFY(0 == stream.read(&data));
			QVERIFY(0 != stream.read(&data));
			QVERIFY(0 != stream.read(&data));
		}
		{
			QFile	f(m_tmp + "/nbody_test/7/stream.idx");
			f.open(QFile::WriteOnly);
			f.write(QByteArray("#coord_size 6\n"));	// Invalid coord size
			f.write(QByteArray("#body_count 256\n"));
			f.write(QByteArray("1\t1\t0\t0\n"));
		}
		{
			nbody_data_stream_reader	stream;
			nbody_data					data;
			data.resize(m_data.get_count());
			QVERIFY(0 != stream.load(m_tmp + "/nbody_test/7/stream"));
		}
		{
			QFile	f(m_tmp + "/nbody_test/7/stream.idx");
			f.open(QFile::WriteOnly);
			f.write(QByteArray("#body_count 256\n")); // Missing coord size
			f.write(QByteArray("1\t1\t0\t0\n"));
		}
		{
			nbody_data_stream_reader	stream;
			nbody_data					data;
			data.resize(m_data.get_count());
			QVERIFY(0 != stream.load(m_tmp + "/nbody_test/7/stream"));
		}
	}


	{
		{
			nbody_data_stream	stream;
			QVERIFY(0 == stream.open(m_tmp + "/nbody_test/8/stream", 1));
			QVERIFY(0 == stream.write(&m_data));
		}

		nbody_data_stream_reader	stream;
		nbody_data					data;
		QVERIFY(0 == stream.load(m_tmp + "/nbody_test/8/stream"));
		QVERIFY(0 != stream.read(NULL));
		QVERIFY(0 != stream.read(&data));
	}
}

class test_nbody_data_io : public QObject
{
	Q_OBJECT
	QString	m_apppath;
public:
	explicit test_nbody_data_io(const QString& apppath):
		m_apppath(QFileInfo(apppath).absolutePath())
	{}
	~test_nbody_data_io() {}
private Q_SLOTS:
	void save_load();
	void load_zeno_ascii();
	void negative();
};

void test_nbody_data_io::save_load()
{
	nbody_data	data1;
	QVERIFY(data1.load(m_apppath + "/../data/initial_state.txt"));
	QVERIFY(data1.save(m_apppath + "/../data/test_save.txt"));

	nbody_data	data2;
	QVERIFY(data2.load(m_apppath + "/../data/test_save.txt"));
	QVERIFY(data2.is_equal(data1, 1e-16));
}

void test_nbody_data_io::load_zeno_ascii()
{
	nbody_data	data1;
	QVERIFY(data1.load_zeno_ascii(m_apppath + "/../data/zeno_ascii.txt"));

	nbody_data	data2;
	QVERIFY(data2.load(m_apppath + "/../data/zeno_table.txt"));
	QVERIFY(data2.is_equal(data1, 1e-16));
}

void test_nbody_data_io::negative()
{
	QVERIFY(QFile::exists(m_apppath + "/../data/inv_column.txt"));
	QVERIFY(!nbody_data().load(m_apppath + "/../data/inv_column.txt"));
	QVERIFY(QFile::exists(m_apppath + "/../data/inv_columns_count.txt"));
	QVERIFY(!nbody_data().load(m_apppath + "/../data/inv_columns_count.txt"));
	QVERIFY(QFile::exists(m_apppath + "/../data/inv_radius.txt"));
	QVERIFY(!nbody_data().load(m_apppath + "/../data/inv_radius.txt"));
}

class test_nbody_stream_restart : public QObject
{
	Q_OBJECT

	const QString		m_tmp;
public:
	test_nbody_stream_restart();
	~test_nbody_stream_restart();
private Q_SLOTS:
	void initTestCase() {}
	void cleanupTestCase() {}
	void run();
};

test_nbody_stream_restart::test_nbody_stream_restart() :
	m_tmp(QDir::tempPath())
{
}

test_nbody_stream_restart::~test_nbody_stream_restart()
{
}

void test_nbody_stream_restart::run()
{
	const QString		expected_stream_name(m_tmp + "/stream-test/restart1_");
	const QString		stream_name(m_tmp + "/stream-test/restart2_");

	const nbcoord_t		box_size = 100;
	const nbcoord_t		max_time = 0.2;

	{
		nbody_data			data;
		nbody_engine_simple	e;
		nbody_solver_euler	s;

		data.make_universe(128, box_size, box_size, box_size);

		QVERIFY(e.init(&data));
		s.set_time_step(0.01, 0.01);
		s.set_engine(&e);
		nbody_data_stream	stream;

		QVERIFY(0 == stream.open(expected_stream_name, 74000));
		QVERIFY(0 == s.run(&data, &stream, max_time, 0.01, 0.1));
		stream.close();
	}
	qDebug() << "Write half";
	{
		nbody_data			data;
		nbody_engine_simple	e;
		nbody_solver_euler	s;

		data.make_universe(128, box_size, box_size, box_size);

		QVERIFY(e.init(&data));
		s.set_time_step(0.01, 0.01);
		s.set_engine(&e);
		nbody_data_stream	half_stream;

		QVERIFY(0 == half_stream.open(stream_name, 74000));
		QVERIFY(0 == s.run(&data, &half_stream, max_time / 2, 0.01, 0.1));
		half_stream.close();
	}
	qDebug() << "Restart";
	{
		nbody_data_stream_reader	append_to;
		nbody_data					data;

		QVERIFY(0 == append_to.load(stream_name));
		QVERIFY(0 == append_to.seek(append_to.get_frame_count() - 1));
		data.resize(append_to.get_body_count());
		QVERIFY(0 == append_to.read(&data));

		nbody_engine_simple	e;
		nbody_solver_euler	s;

		QVERIFY(e.init(&data));
		s.set_time_step(0.01, 0.01);
		s.set_engine(&e);
		nbody_data_stream			half_stream;
		QVERIFY(0 == half_stream.open(stream_name, 74000, &append_to));
		append_to.close();
		QVERIFY(0 == s.run(&data, &half_stream, max_time, 0.01, 0.1));
		half_stream.close();
	}

	{
		nbody_data_stream_reader	expected_stream;
		nbody_data_stream_reader	stream;
		nbody_data					expected_data;
		nbody_data					data;

		QVERIFY(0 == expected_stream.load(expected_stream_name));
		QVERIFY(0 == stream.load(stream_name));

		data.resize(stream.get_body_count());
		expected_data.resize(expected_stream.get_body_count());

		QVERIFY(expected_stream.get_frame_count() == stream.get_frame_count());
		size_t frame_count = std::min(expected_stream.get_frame_count(), stream.get_frame_count());

		for(size_t frame_n = 0; frame_n < frame_count; ++frame_n)
		{
			qDebug() << frame_n;
			QVERIFY(0 == expected_stream.seek(frame_n));
			QVERIFY(0 == stream.seek(frame_n));
			QVERIFY(0 == expected_stream.read(&expected_data));
			QVERIFY(0 == stream.read(&data));
			QVERIFY(expected_data.is_equal(data, 1e-16));
		}
	}
}

int main(int argc, char* argv[])
{
	int	res = 0;
	{
		test_nbody_stream tc1;
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		test_nbody_data_io tc1(argv[0]);
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		test_nbody_stream_restart tc1;
		res += QTest::qExec(&tc1, argc, argv);
	}
	return res;
}

#include "test_nbody_stream.moc"
