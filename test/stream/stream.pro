include( ../../pri/nbody.pri )
include( ../../pri/opencl.pri )
include( ../../pri/vectorize.pri )

TEMPLATE	= app
TARGET		= stream
MOC_DIR = ./.tmp/moc

CONFIG		+= qt testcase
QT += opengl testlib
LIBS += -lGLU

INCLUDEPATH += ../../nbody
LIBS += -L../../nbody
LIBS += -lnbody

SOURCES += test_nbody_stream.cpp
