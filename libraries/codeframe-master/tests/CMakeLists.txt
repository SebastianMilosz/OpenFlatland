cmake_minimum_required(VERSION 3.8)

project (CODEFRAME_TESTS VERSION 1.0.0)

add_executable(CODEFRAME_TESTS
        catch.hpp
        test_serializable_container.cpp
        test_serializable_path.cpp
        test_serializable_storage.cpp
        test_serializable_utilities.cpp
        test_serializablelibmain.cpp
        ../../../src/ray_data.cpp
    )

set_target_properties(CODEFRAME_TESTS PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_link_libraries( CODEFRAME_TESTS
    codeframe 
    lualib
    pthread
)