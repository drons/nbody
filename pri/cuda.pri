# Try to detect Nvidia SDK
exists( /usr/local/cuda/bin ){
	LIBS += -L/usr/local/cuda/lib64
	LIBS += -L/usr/local/cuda/lib64/stubs
	INCLUDEPATH += /usr/local/cuda/include
	QMAKE_CUC = /usr/local/cuda/bin/nvcc

	LIBS += -lcuda
	LIBS += -lcudart
	DEFINES += HAVE_CUDA
	#QMAKE_CUDA_SM=-gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_61,code=sm_61

	# CUDA compiler rule
	cu.name = Cuda ${QMAKE_FILE_IN}
	cu.input = CUSOURCES
	cu.CONFIG += no_link
	cu.variable_out = OBJECTS

	isEmpty(CU_DIR):CU_DIR = .
	isEmpty(QMAKE_CPP_MOD_CU):QMAKE_CPP_MOD_CU = 
	isEmpty(QMAKE_EXT_CPP_CU):QMAKE_EXT_CPP_CU = .cu

	QMAKE_CUEXTRAFLAGS += -m64
	!greaterThan(COMPILER_MAJOR_VERSION, 6){
		QMAKE_CUEXTRAFLAGS += -std=c++11
	}
	greaterThan(COMPILER_MAJOR_VERSION, 6){
		QMAKE_CUEXTRAFLAGS += -std=c++14
	}
	greaterThan(COMPILER_MAJOR_VERSION, 8){
		QMAKE_CXXFLAGS += -Wno-old-style-cast
	}
	QMAKE_CUEXTRAFLAGS += --compile
	QMAKE_CUEXTRAFLAGS += $$join(INCLUDEPATH,'" -I"','-I"','"')
	QMAKE_CUEXTRAFLAGS += $$join(DEFINES,'" -D"','-D"','"')
	QMAKE_CUEXTRAFLAGS += -Xcompiler $$escape_expand(\")-fPIC$$escape_expand(\")
	QMAKE_CUEXTRAFLAGS += -lineinfo

	contains( DEFINES, DEBUG ){
			QMAKE_CUEXTRAFLAGS += --debug
	}

	CUDA_OBJECTS_DIR = ./

	cu.commands = $$QMAKE_CUC $$QMAKE_CUEXTRAFLAGS -o $$CUDA_OBJECTS_DIR/$${QMAKE_CPP_MOD_CU}${QMAKE_FILE_BASE}$${QMAKE_EXT_OBJ} ${QMAKE_FILE_NAME}$$escape_expand(\n\t)
	cu.output = $$CUDA_OBJECTS_DIR/$${QMAKE_CPP_MOD_CU}${QMAKE_FILE_BASE}$${QMAKE_EXT_OBJ}
	silent:cu.commands = @echo nvcc ${QMAKE_FILE_IN} && $$cu.commands
	QMAKE_EXTRA_COMPILERS += cu

	build_pass|isEmpty(BUILDS):cuclean.depends = compiler_cu_clean
	else:cuclean.CONFIG += recursive
	QMAKE_EXTRA_TARGETS += cuclean
}
