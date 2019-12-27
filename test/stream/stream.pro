include( ../../pri/nbody.pri )

TEMPLATE	= app
TARGET		= stream
MOC_DIR = ./.tmp/moc
DESTDIR = ./

CONFIG		+= qt testcase
QT += opengl testlib
win32:LIBS += -lGLU32
unix:LIBS += -lGLU

INCLUDEPATH += ../../nbody
LIBS += -L../../lib
LIBS += -lnbody

SOURCES += test_nbody_stream.cpp

TEST_DATA += \
	../data/inv_column.txt \
	../data/inv_columns_count.txt \
	../data/inv_radius.txt \
	../data/zeno_ascii.txt \
	../data/zeno_table.txt
include( ../../pri/testdata.pri )
