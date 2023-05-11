include( ../../pri/nbody.pri )

TEMPLATE	= app
TARGET		= nbody-simulation
MOC_DIR = ./.tmp/moc
DESTDIR = ./

CONFIG		+= qt

INCLUDEPATH += ../../nbody
LIBS += -L../../lib
LIBS += -lnbody

SOURCES	+= main.cpp

