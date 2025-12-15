cmake_minimum_required(VERSION 3.21)
project(RNN_p)

# Use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler optimization flags
add_compile_options(-O2)
#add_compile_options(-Ofast)

# Include headers
include_directories(src)

# Gather all .cpp files
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
        src/*.cpp
)

# Define executable
add_executable(RNN_p ${SOURCES})
