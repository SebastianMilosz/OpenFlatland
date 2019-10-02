cmake_minimum_required(VERSION 2.8.11)

project (CODEFRAME_LIBSRC)

add_library(CODEFRAME_LIBSRC
    base64.hpp
    cxml.hpp
    instancemanager.hpp
    serializable.hpp
    serializableinterface.hpp
    serializablecontainer.hpp
    serializableproperty.hpp
    xmlformatter.hpp
    serializablechildlist.hpp
    serializablestreammanager.h
    qtserializableutilities.h
    rs485protocol.h
    communicationinterface.h
    wxserializable.h
)