 
CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
PROJECT( m-slam ) #project name

#compiler
#SET(CMAKE_CXX_COMPILER "g++")

#build a debug
#SET( CMAKE_BUILD_TYPE Debug  )

#Put compiled files with main functions here (executable)
#SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

#Put compiled library files here(executable, helper functions)
#SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)


SET(CMAKE_CXX_COMPILER "g++")
SET( CMAKE_BUILD_TYPE Release  )
# SET( CMAKE_CXX_FLAGS "-std=c++11 -g -O3")   # non optimized for gdb debug
SET( CMAKE_CXX_FLAGS "-std=c++11 -O3")
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#Link the directory
LINK_DIRECTORIES( ${PROJECT_SOURCE_DIR}/lib)

#Save header files here
INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/include )

#Will build this as part of the project too.
#Should contain cmake file too.
ADD_SUBDIRECTORY( ${PROJECT_SOURCE_DIR}/src )
