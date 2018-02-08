cmake_minimum_required(VERSION 2.8.11)

project (CODEFRAME_LIBSRC)

add_library(CODEFRAME_LIBSRC
        cxml.h
        instancemanager.h
        qtserializableutilities.h
        serializable.h
        serializableinterface.h
        serializablecontainer.h
        serializableproperty.h
        serializablestreammanager.h
        xmlformatter.h
        rs485protocol.h
        communicationinterface.h
        wxserializable.h
        serializablechildlist.h
		xmlformatter.cpp
		serializableregister.cpp
		serializableproperty.cpp
		serializablecontainer.cpp
		serializableinterface.cpp
		serializable.cpp
		instancemanager.cpp
		cxml.cpp
		serializablelua.cpp
		rs485protocol.cpp
		serializableversion.cpp
		wxserializable.cpp
		serializablechildlist.cpp		
	)