
exists( /opt/intel/opencl/include ){
	LIBS += -L/opt/intel/opencl
	INCLUDEPATH += /opt/intel/opencl/include
	DEFINES += HAVE_OPENCL
}

!contains( DEFINES, HAVE_OPENCL ){
	exists( /home/sas/prg/opencl/nvidia ){
		LIBS += -L/home/sas/prg/opencl/nvidia/lib
		INCLUDEPATH += /home/sas/prg/opencl/nvidia
		DEFINES += HAVE_OPENCL
	}
}

contains( DEFINES, HAVE_OPENCL ){
	LIBS += -lOpenCL
}
