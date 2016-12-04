include( ../pri/nbody.pri )
include( ../pri/opencl.pri )
include( ../pri/vectorize.pri )

TEMPLATE	= app
TARGET		= nbodyplayer
MOC_DIR = ./.tmp/moc

CONFIG		+= qt
QT += opengl
LIBS += -lGLU -lgomp

INCLUDEPATH += ../nbody
LIBS += -L../nbody
LIBS += -lnbody

SOURCES	+= main.cpp \
    wgt_nbody_view.cpp \
    wgt_nbody_player.cpp \
    wgt_nbody_player_control.cpp \
    nbody_frame_compressor.cpp \
    nbody_frame_compressor_image.cpp \
    nbody_frame_compressor_av.cpp

HEADERS	+= \
	wgt_nbody_view.h \
    wgt_nbody_player.h \
    wgt_nbody_player_control.h \
    nbody_frame_compressor.h \
    nbody_frame_compressor_image.h \
    nbody_frame_compressor_av.h



