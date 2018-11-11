#include <QString>
#include <QtTest>

class test_memsupp : public QObject
{
	Q_OBJECT

public:
	test_memsupp();
	~test_memsupp();
private Q_SLOTS:
	void initTestCase();
	void cleanupTestCase();
	void test1();
	void test2();
};

test_memsupp::test_memsupp()
{
}

test_memsupp::~test_memsupp()
{
}

void test_memsupp::initTestCase()
{
	QVERIFY(true);
}

void test_memsupp::cleanupTestCase()
{
}

void test_memsupp::test1()
{
	qDebug() << "Generate Valgrind suppressions origin";
}

void test_memsupp::test2()
{
	//Suppress error at QLocale::QLocale() from QString::toDouble()
	QString	val("+4.2820153172908482e-03");
	qDebug() << val.toDouble();
}

int main(int argc, char* argv[])
{
	int res = 0;

	{
		test_memsupp tc1;
		res += QTest::qExec(&tc1, argc, argv);
	}

	return res;
}

#include "test_memsupp.moc"
