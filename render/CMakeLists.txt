cmake_minimum_required(VERSION 3.16)
project(motion_ctr)

add_compile_options(-fPIC)
add_compile_options(-std=gnu++11)
add_compile_options(-Wdeprecated-declarations)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_BUILD_TYPE Release CACHE STRING
	"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
	)
set(CMAKE_AUTOMOC ON)


link_directories(../sim/)
include_directories(../sim/)
link_directories(../network/)
include_directories(../network/)

find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Qt5Widgets CONFIG REQUIRED)
find_package(Qt5OpenGL CONFIG REQUIRED)
link_libraries(stdc++fs)

add_compile_options(-DHAVE_CSTDDEF)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${DART_INCLUDE_DIRS})
include_directories(${Pybind11_INCLUDE_DIRS})
include_directories(${TinyXML_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_DIR})


include(FindOpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

file(GLOB SOURCES *.cpp *.h)
add_executable(motion_ctr ${SOURCES})
target_link_libraries(motion_ctr Qt5::Widgets Qt5::OpenGL 
	${DART_LIBRARIES} ${Pybind11_LIBRARIES} ${PYTHON_LIBRARIES} GL GLU glut GLEW sim ${TinyXML_LIBRARIES} 
	
)
