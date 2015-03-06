#-------------------------------------------------------
#
# Project created by Caner Hazirbas 2014-11-08T10:05:00
#
#-------------------------------------------------------

TEMPLATE = app
CONFIG += console
CONFIG -= qt
DEFINES += QMLJSDEBUGGER

# Install target (AFS) to the main project folder
target.path = .
INSTALLS += target

# Clean command
QMAKE_CLEAN += -r "./../../../AFS"

INCLUDEPATH += include/

HEADERS += \
    include/main.hpp \
    include/common.hpp \
    include/Image.hpp \
    include/Dataset.hpp \
    include/Feature.hpp \
    include/ImageProc.hpp \
    include/Model.hpp \
    include/Learning.hpp \
    include/Potential.hpp \
    include/Optimization.hpp \
    cuda/segmentation.cuh \
    cuda/binarization.cuh \
    cuda/cutil5.cuh

SOURCES += \
    src/main.cpp \
    src/Image.cpp \
    src/Dataset.cpp \
    src/ImageProc.cpp \
    src/Model.cpp \
    src/Potential.cpp \
    src/Learning.cpp \
    src/Feature.cpp \
    src/Optimization.cpp \
    misc/edgeDetectionFct.cpp

# List CUDA source files on the filetree in QtCreator
OTHER_FILES += \
    cuda/segmentation.cu \
    cuda/binarization.cu \


# Compiler flags tuned for my system
QMAKE_CXXFLAGS += -I../ \
    -O99 \
    -pipe \
    -g \
#    -Wall
    -w

LIBS += -L../ \
    -L/usr/lib \
    -lcuda \
    -lcudart \
    -lm \
    -lX11

# OpenCV Libs

INCLUDEPATH += /usr/wiss/hazirbas/Libs/OpenCV/2.4.10/include

LIBS += -L/usr/wiss/hazirbas/Libs/OpenCV/2.4.10/lib/ \
        -lopencv_core \
        -lopencv_imgproc \
        -lopencv_highgui \
        -lopencv_ml \
        -lopencv_video \
        -lopencv_features2d \
        -lopencv_nonfree \
        -lopencv_calib3d \
        -lopencv_objdetect \
        -lopencv_contrib \
        -lopencv_legacy \
        -lopencv_flann \

# Boost Libs

INCLUDEPATH += /usr/wiss/hazirbas/Libs/Boost/1.54/build/include

LIBS += -L/usr/wiss/hazirbas/Libs/Boost/1.54/build/lib \
        -lboost_system \
        -lboost_filesystem


# #######################################################################
# CUDA
# #######################################################################
# auto-detect CUDA path

CUDA_ARCH     = sm_35
CUDA_DIR      = /usr/local/cuda-6.5

INCLUDEPATH += $$CUDA_DIR/include

LIBS += -L$$CUDA_DIR/lib64/ \
        -lcudart

CUDA_SOURCES += cuda/segmentation.cu \
                cuda/binarization.cu \

cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.obj
cuda.commands = $$CUDA_DIR/bin/nvcc \
    -arch $$CUDA_ARCH \
    -c \
    $$NVFLAGS \
    -Xptxas -v \
    -Xcompiler \
    $$join(QMAKE_CXXFLAGS,",") \
    $$join(INCLUDEPATH,'" -I "','-I "','"') \
    ${QMAKE_FILE_NAME} \
    -o \
    ${QMAKE_FILE_OUT}
cuda.dependcy_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc \
    -M \
    -arch $$CUDA_ARCH \
    -Xptxas -v \
    -Xcompiler \
    $$join(QMAKE_CXXFLAGS,",") \
    $$join(INCLUDEPATH,'" -I "','-I "','"') \
    ${QMAKE_FILE_NAME} \
    | \
    tr \
    -d \
    '\\\n'
cuda.input = CUDA_SOURCES
QMAKE_EXTRA_COMPILERS += cuda

RESOURCES +=
