# AFS
Automatic Feature Selection is a code framework for the feature selection method introduced in 

>Caner Hazirbas, Julia Diebold, Daniel Cremers,
>**Optimizing the Relevance-Redundancy Tradeoff for Efficient Semantic Segmentation**,
>*In Scale Space and Variational Methods in Computer Vision*, 2015.

## Framework Structure
    --AFS    
        |
        -- build            ; build directories for CMake and QtCreator projects
        |
        -- cuda             ; cuda source/header files
        |
        -- doc              ; code documentation and related conference paper
            |
            -- html         ; html documentation (double-click on **index.html**)
            |
            -- latex        ; PDF documentation             
        |
        -- include          ; header files
        |
        -- misc             ; miscellaneous source files and mrmr
            |
            -- mrmr         ; folder to store mrmr executable file
        |
        -- src              ; source files
        |
        -- AFS.pro          ; QtCreator Project File
        |
        -- CMakeLists.txt   ; CMake Project File

## How To Build
### Required Hardware/Software

To be able to compile and run the code you need a computer with

* Ubuntu OS (12.04/14.04)
* NVidia GPU with CUDA support

You need to install the following libraries (I recommend you to download and compile the libraries from source):

* [OpenCV 2.4.10](http://opencv.org/downloads.html)
* [Boost  1.54.0](http://www.boost.org/users/history/version_1_54_0.html)

Along with these libraries, you need to install CUDA drivers on your machine:

* [CUDA 6.5 Production Release](https://developer.nvidia.com/cuda-downloads)

Once libraries are installed, you can download the source from github:

            git clone https://github.com/tum-vision/AFS.git
     
This framework requires [mrmr](http://penglab.janelia.org/proj/mRMR/) method to be compiled. Please download the [source files](http://penglab.janelia.org/proj/mRMR/mrmr_c_src.zip) inside the folder *misc/mrmr/* and compile the code with **make** command.

### Build from CMake

1. Please export the following paths in your **.bashrc** **if you built libraries from source**. 
    * OpenCV 2.4.10
    ```
    export OPENCV_DIR=<path to OpenCV>/OpenCV/2.4.10/share/OpenCV
    export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:<path to OpenCV>/OpenCV/2.4.10/lib/pkgconfig
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path to OpenCV>/OpenCV/2.4.10/lib  
    ```
    * Boost 1.54.0
    ```    
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path to Boost>/Boost/1.54/build/lib
    ```

2. Change the following lines in CMakeLists.txt
    *   **Line 13** : Set path to the **OpenCVConfig.cmake**
    *   **Line 23** : Set **CUDA Compute Capability** in arch=compute_??,code=sm_??
    
3. Run console in AFS/build/CMake and type:
    ```
    cmake ../../
    make 
    make install
    make clean (to clean the project)
    ```
### Build from QtCreator(qmake)

1. Set *INCLUDEPATH* and *LIBS* path in **AFS.pro** (qmake project file) for :
    * OpenCV 2.4.10: **Lines 71, 73**
    * Boost 1.54.0 : **Lines 89, 91**
    * CUDA 6.5     : **Lines 104, 106**
2. Set CUDA installation directory(**CUDA_DIR**) and CUDA compute capability(**CUDA_ARCH**) in **Lines 101, 102**

3. You can use build/QtCreator/Debug and build/QtCreator/Release folders to compile the project with QtCreator

Both builds will copy the executable file (*AFS*) into main project folder. 
You can run the executable as **./AFS**.