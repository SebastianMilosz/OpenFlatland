cmake_minimum_required(VERSION 2.8.11)

project (CODEFRAME_THIRD_PARTY_LIB)

add_library(CODEFRAME_THIRD_PARTY_LIB
		pugixml-1.7/src/pugixml.cpp
		pugixml-1.7/src/pugiconfig.hpp
        pugixml-1.7/src/pugixml.hpp
        sigslot-1.0.0/sigslot.h
        smartpointer/smartpointer.h
	)