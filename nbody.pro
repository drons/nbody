include( pri/nbody.pri )

TEMPLATE = subdirs

SUBDIRS += nbody

!contains(DEFINES,NB_COORD_PRECISION=4){
    !contains(CONFIG, NO_UI){
        SUBDIRS += player
        player.subdir = player
        player.depends = nbody
    }
}

SUBDIRS += test
test.subdir = test
test.depends = nbody

OTHER_FILES += \
	appveyor.yml \
	.astylerc \
	.github/workflows/ubuntu-gpu-build.sh \
	.github/workflows/ubuntu-gpu-build.yml \
	README.md \
	.travis.yml

OTHER_FILES += test/pvs/plog.cfg
