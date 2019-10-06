cmake_minimum_required(VERSION 2.8.11)

project (CODEFRAME_THIRD_PARTY_LIB)

set(CODEFRAME_THIRD_PARTY_HDRS
    pugixml-1.7/src/pugiconfig.hpp
    pugixml-1.7/src/pugixml.hpp
    sigslot-1.0.0/sigslot.h
    smartpointer/smartpointer.h
)

set(CODEFRAME_THIRD_PARTY_SRC
    pugixml-1.7/src/pugixml.cpp
)

include_directories( pugixml-1.7/src )

add_library(CODEFRAME_THIRD_PARTY_LIB STATIC
    ${CODEFRAME_THIRD_PARTY_HDRS}
    ${CODEFRAME_THIRD_PARTY_SRC}
)