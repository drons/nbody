include( ../../pri/nbody.pri )

TEMPLATE	= app
TARGET		= memsupp
MOC_DIR = ./.tmp/moc
DESTDIR = ./

CONFIG		+= qt testcase
QT += opengl testlib
win32:LIBS += -lGLU32
unix:LIBS += -lGLU

INCLUDEPATH += ../../nbody
LIBS += -L../../lib
LIBS += -lnbody

SOURCES += test_memsupp.cpp

OTHER_FILES += memcheck.supp
