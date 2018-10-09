win32{
	contains( DEFINES, DEBUG ){
		OPEN_CV_LIB_SUFFIX="d"
	}
}

OPEN_CV_VERSION=

LIBS += -lopencv_flann$${OPEN_CV_VERSION}$${OPEN_CV_LIB_SUFFIX}
LIBS += -lopencv_imgproc$${OPEN_CV_VERSION}$${OPEN_CV_LIB_SUFFIX}
LIBS += -lopencv_features2d$${OPEN_CV_VERSION}$${OPEN_CV_LIB_SUFFIX}
LIBS += -lopencv_highgui$${OPEN_CV_VERSION}$${OPEN_CV_LIB_SUFFIX}
LIBS += -lopencv_legacy$${OPEN_CV_VERSION}$${OPEN_CV_LIB_SUFFIX}
LIBS += -lopencv_core$${OPEN_CV_VERSION}$${OPEN_CV_LIB_SUFFIX}
LIBS += -lopencv_video$${OPEN_CV_VERSION}$${OPEN_CV_LIB_SUFFIX}
LIBS += -lopencv_contrib$${OPEN_CV_VERSION}$${OPEN_CV_LIB_SUFFIX}
LIBS += -lopencv_superres$${OPEN_CV_VERSION}$${OPEN_CV_LIB_SUFFIX}

