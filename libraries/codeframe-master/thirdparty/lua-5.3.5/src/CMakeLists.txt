cmake_minimum_required (VERSION 2.6)
project (lua)  # project here actually means solution in premake

if(WIN32)
        add_definitions( -D_CRT_SECURE_NO_WARNINGS )
endif()

# 1. lua static library
# how to rename library name?
add_library (lualib STATIC lapi.c lauxlib.c lbaselib.c lbitlib.c lcode.c lcorolib.c lctype.c ldblib.c ldebug.c ldo.c ldump.c lfunc.c lgc.c linit.c liolib.c llex.c lmathlib.c lmem.c loadlib.c lobject.c lopcodes.c loslib.c lparser.c lstate.c lstring.c lstrlib.c ltable.c ltablib.c ltm.c lua.c luac.c lundump.c lutf8lib.c lvm.c lzio.c)
set_target_properties(lualib PROPERTIES OUTPUT_NAME "lua") # rename the library output name

# 2. lua interpreter
link_directories (${LUA_BINARY_DIR})
add_executable (lua lua.c)
target_link_libraries (lua lualib)
if(UNIX)
        target_link_libraries( lua m )
endif()

# 3. lua compiler
link_directories (${LUA_BINARY_DIR})
add_executable (luac luac.c)
target_link_libraries (luac lualib)
if(UNIX)
        target_link_libraries( luac m )
endif()
