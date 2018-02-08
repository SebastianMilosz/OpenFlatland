cmake_minimum_required(VERSION 2.8.11)

project (CODEFRAME_TESTS)

add_library(CODEFRAME_TESTS
		catch.hpp
		test_serializable.cpp
		test_serializablelibmain.cpp
	)