QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    benchmark.cpp \
    display.cpp \
    main.cpp \
    refRenderer.cpp \
    sceneLoader.cpp

HEADERS += \
    badCudaRenderer.h \
    circleRenderer.h \
    cudaRenderer.h \
    cycleTimer.h \
    image.h \
    refRenderer.h \
    sceneLoader.h

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

DISTFILES += \
    cudaRenderer.cu \
    badCudaRenderer.cu \
    exclusiveScan.cu_inl \
    circleBoxTest.cu_inl \
    CUDA_Circle_Renderer.pro.user

# CUDA
# Path to cuda toolkit install
CUDA_DIR      = /opt/cuda
# GPU architecture (ADJUST FOR YOUR GPU)
CUDA_GENCODE  = arch=compute_61,code=sm_61
# manually add CUDA sources (ADJUST MANUALLY)
CUDA_SOURCES += \
    cudaRenderer.cu \
    badCudaRenderer.cu
# Path to header and libs files
INCLUDEPATH  += $$CUDA_DIR/include
# libs used in your code
LIBS         += -L $$CUDA_DIR/lib64 -lcudart -lcuda -lglut -lGL

cuda.commands        = $$CUDA_DIR/bin/nvcc -c -gencode $$CUDA_GENCODE -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}

cuda.dependency_type = TYPE_C
cuda.depend_command  = $$CUDA_DIR/bin/nvcc -M ${QMAKE_FILE_NAME} | sed \"s/^.*: //\"
cuda.input           = CUDA_SOURCES
cuda.output          = ${QMAKE_FILE_BASE}_cuda.o
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda
