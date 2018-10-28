# Try to detect default installation from 'ocl-icd-opencl-dev' package with OpenCL 2 cpp API
exists( /usr/include/CL/cl2.hpp ){
	DEFINES += HAVE_OPENCL
	DEFINES += HAVE_OPENCL2
}

# Try to detect default installation from 'ocl-icd-opencl-dev' package with OpenCL old cpp API
!contains( DEFINES, HAVE_OPENCL ){
	exists( /usr/include/CL/cl.hpp ){
		DEFINES += HAVE_OPENCL
	}
}

# Try to detect Intel SDK
!contains( DEFINES, HAVE_OPENCL ){
	exists( /opt/intel/opencl/include ){
		LIBS += -L/opt/intel/opencl
		INCLUDEPATH += /opt/intel/opencl/include
		DEFINES += HAVE_OPENCL
	}
}

# Try to detect Nvidia SDK
!contains( DEFINES, HAVE_OPENCL ){
	exists( /usr/local/cuda/ ){
		LIBS += -L/usr/local/cuda/libs64
		INCLUDEPATH += /usr/local/cuda/include
		DEFINES += HAVE_OPENCL
	}
}

contains( DEFINES, HAVE_OPENCL ){
	LIBS += -lOpenCL
}
