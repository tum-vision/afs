# CMakeLists.txt for Automatic Feature Selection (AFS)
# created by Caner Hazirbas, 27.02.2015 15:58
# Technische Universität München
# caner.hazirbas@cs.tum.edu
 
project( AFS )

# Required CMAKE Version
cmake_minimum_required( VERSION 2.8 )

# SET LIBRARY DIRECTORIES
# OpenCV
set(OpenCV_DIR "/usr/wiss/hazirbas/Libs/OpenCV/2.4.10/share/OpenCV")

find_package( OpenCV REQUIRED )
find_package( Boost  REQUIRED COMPONENTS system filesystem )
find_package( CUDA   REQUIRED )

# list all files
include_directories(. include src cuda misc)

# set cuda flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_35,code=sm_35 -Xptxas -v -Xcompiler )

# add_executable
cuda_add_executable(AFS

			src/Image.cpp
			src/KMeansClustering.cpp
			src/main.cpp
			src/Optimization.cpp
			src/Dataset.cpp
			src/Feature.cpp
			src/ImageProc.cpp
			src/Learning.cpp
			src/Model.cpp
			src/Potential.cpp
			cuda/binarization.cu
			cuda/segmentation.cu
			misc/edgeDetectionFct.cpp

			include/common.hpp
			include/Dataset.hpp
			include/Feature.hpp
			include/Image.hpp
			include/ImageProc.hpp
			include/Learning.hpp
			include/main.hpp
			include/Model.hpp
			include/Optimization.hpp
			include/Potential.hpp
			cuda/binarization.cuh
			cuda/cutil5.cuh
			cuda/segmentation.cuh
)

# link libraries
target_link_libraries(AFS ${OpenCV_LIBS} ${Boost_LIBRARIES})

# install AFS to the project folder
SET(CMAKE_INSTALL_PREFIX ./)
install(TARGETS AFS DESTINATION ../../)

# redefine 'clean' for make
set_directory_properties(PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ../../AFS) 
