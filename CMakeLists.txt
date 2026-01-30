cmake_minimum_required(VERSION 3.12)
project(ASTen LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Debug)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Optional CUDA support
option(USE_CUDA "Enable CUDA support" OFF)
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-D__CUDA__)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# C10 library (core)
set(C10_SOURCES
    c10/core/storage.c
    c10/core/tensor.c
)

add_library(c10 STATIC ${C10_SOURCES})

# aten library ie ops
set(ATEN_SOURCES
        aten/native/view_ops.c
)
add_library(aten STATIC ${ATEN_SOURCES})
target_link_libraries(aten c10)


if(USE_CUDA)
    set(CUDA_SOURCES
    )

    add_library(aten_cuda STATIC ${CUDA_SOURCES})
    target_link_libraries(aten_cuda c10)
    set_target_properties(aten_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Python bindings
pybind11_add_module(_C ASTen/bindings.cpp)
target_link_libraries(_C PRIVATE c10 aten)


if(USE_CUDA)
    target_link_libraries(_C PRIVATE aten_cuda ${CUDA_LIBRARIES})
endif()


# Installation
install(TARGETS _C LIBRARY DESTINATION ASTen)
install(DIRECTORY ASTen/ DESTINATION ASTen
        FILES_MATCHING PATTERN "*.py")