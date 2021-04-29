cmake_minimum_required(VERSION 3.16)
project(SimEnv)

add_compile_options(-fPIC)
add_compile_options(-std=gnu++11)
add_compile_options(-Wdeprecated-declarations)


link_directories(../sim/)
include_directories(../sim/)

file(GLOB srcs "*.h" "*.cpp" "*.hpp")
include_directories(${Pybind11_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${DART_INCLUDE_DIRS})
include_directories(${TinyXML_INCLUDE_DIRS})


include(FindOpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
#add_library(SimEnv ${srcs})

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})



pybind11_add_module(SimEnv ${srcs})
target_link_libraries(SimEnv PRIVATE ${DART_LIBRARIES} ${Pybind11_LIBRARIES} ${TinyXML_LIBRARIES} ${PYTHON_LIBRARIES} sim)
# set_target_properties(SimEnv PROPERTIES PREFIX "" )
