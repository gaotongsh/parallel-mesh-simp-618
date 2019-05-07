cmake_minimum_required(VERSION 2.8)

# Application source
set(APPLICATION_SOURCE
    main.cpp
   
    halfEdgeMesh.cpp
    meshEdit.cpp

    offFileManager.cpp

    matrix3x3.cpp
    matrix4x4.cpp
    vector3D.cpp
    vector4D.cpp

    cuda_simplify/cycleTimer.c
    cuda_simplify/cycleTimer.h
    cuda_simplify/gpuSimplify.cpp
    cuda_simplify/gpuSimplify.h
    )

set(CUDA_SEPARABLE_COMPILATION ON)
list(APPEND CUDA_NVCC_FLAGS
  -arch=sm_61 -std=c++11)
list(APPEND CUDA_NVCC_FLAGS
  "-Xcompiler ,\"-std=c++11\"")

CUDA_ADD_LIBRARY( cuda_simplify
        cuda_simplify/cudaSimplify.cu
        cuda_simplify/cudaScan.cu
        cuda_simplify/cudaSort.cu
)

#-------------------------------------------------------------------------------
# Add executable
#-------------------------------------------------------------------------------
add_executable(parallel-mesh-simp ${APPLICATION_SOURCE})

target_compile_options(parallel-mesh-simp PRIVATE "-O3" "-march=native")

target_link_libraries(parallel-mesh-simp
    cuda_simplify
    ${CMAKE_THREADS_INIT}
)

#-------------------------------------------------------------------------------
# Platform-specific configurations for target
#-------------------------------------------------------------------------------
if(APPLE)
  set_property( TARGET parallel-mesh-simp APPEND_STRING PROPERTY COMPILE_FLAGS
                "-Wno-deprecated-declarations -Wno-c++11-extensions")
endif(APPLE)

# Put executable in build directory root
set(EXECUTABLE_OUTPUT_PATH ..)

# Install to project root
install(TARGETS parallel-mesh-simp DESTINATION .)
