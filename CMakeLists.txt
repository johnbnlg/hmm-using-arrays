cmake_minimum_required(VERSION 3.27)
project(hmm C)

set(CMAKE_C_STANDARD 90)

add_library(hmm hmm.c)
target_link_libraries(hmm m)

add_library(hmm_mpi hmm_mpi.c)
target_link_libraries(hmm_mpi hmm)

if (${WIN32})
    target_link_libraries(hmm_mpi msmpi)
    target_include_directories(hmm_mpi PRIVATE $ENV{MSMPI_INC})
    target_link_directories(hmm_mpi PRIVATE $ENV{MSMPI_LIB64})
else ()
endif ()
