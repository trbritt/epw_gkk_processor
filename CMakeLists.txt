cmake_minimum_required(VERSION 3.20.0)
project(process_g)

find_package(MPI REQUIRED COMPONENTS C CXX)

set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp -fbounds-check")
add_compile_options(-O3 -march=native -fopenmp -fbounds-check)
find_package(HDF5 REQUIRED COMPONENTS C HL)
find_package(Boost REQUIRED COMPONENTS  atomic mpi serialization filesystem regex)

# Add the source files to the project
set(SOURCES
    process_g.cpp
)

# Add your project's executable target
add_executable(process_g.x ${SOURCES})

# Include directories
target_include_directories(process_g.x PRIVATE ${HDF5_INCLUDE_DIRS} ${HDF5_C_INCLUDE_DIRS} ${BOOST_INCLUDE_DIRS})

# Link directories
target_link_directories(process_g.x PRIVATE ${BOOST_LIBRARY_DIRS})

# Link libraries
target_link_libraries(process_g.x PRIVATE ${HDF5_C_LIBRARIES} Boost::regex Boost::mpi Boost::serialization)

