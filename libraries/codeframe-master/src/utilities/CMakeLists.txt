cmake_minimum_required(VERSION 2.8.11)

project (CODEFRAME_LIBSRC)

add_library(CODEFRAME_LIBSRC
            utilities.cpp
            FileUtilities.cpp
            LoggerUtilities.cpp
            PerformanceUtilities.cpp
            SysUtilities.cpp
            ThreadUtilities.cpp
            utilities.h
            crc.h
            HashUtilities.h
            MathUtilities.h
            DataTypesUtilities.h
            FilepathUtilities.h
            FileUtilities.h
            LoggerUtilities.h
            PerformanceUtilities.h
            SysUtilities.h
            TextUtilities.h
            ThreadUtilities.h
)