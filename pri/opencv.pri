win32{
	INCLUDEPATH += C:/projects/opencv/include
	LIBS += -LC:/projects/opencv/x64/vc14/lib
	contains( DEFINES, DEBUG ){
		OPEN_CV_LIB_SUFFIX="2412d"
	}
	!contains( DEFINES, DEBUG ){
		OPEN_CV_LIB_SUFFIX="2412"
	}
}

LIBS += -lopencv_highgui$${OPEN_CV_LIB_SUFFIX}
LIBS += -lopencv_core$${OPEN_CV_LIB_SUFFIX}
LIBS += -lopencv_video$${OPEN_CV_LIB_SUFFIX}
greaterThan(OPEN_CV_VERSION_MAJOR, 2){
	LIBS += -lopencv_videoio$${OPEN_CV_LIB_SUFFIX}
}
greaterThan(OPEN_CV_VERSION_MAJOR, 3){
	INCLUDEPATH += /usr/include/opencv4
}
