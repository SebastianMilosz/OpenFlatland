cmake_minimum_required(VERSION 2.8.11)

project (CODEFRAME_LIBSRC)

add_library(CODEFRAME_LIBSRC
    base64.cpp
    cxml.cpp  
    cxmlnode.cpp
    extpoint2d.cpp
    extthrust.cpp
    extvector.cpp
    instancemanager.cpp
    referencemanager.cpp
    serializable.cpp
    serializablechildlist.cpp
    serializablecontainer.cpp
    serializableidentity.cpp
    serializableinterface.cpp
    serializablelua.cpp
    serializablepath.cpp
    serializableproperty.cpp
    serializablepropertyinfo.cpp
    serializablepropertyiterator.cpp
    serializablepropertymanager.cpp
    serializableregister.cpp
    serializableselectable.cpp
    serializablestorage.cpp
    serializablestream.cpp
    serializablestreammanager.cpp
    typeinfo.cpp
    xmlformatter.cpp
)