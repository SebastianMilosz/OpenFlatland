#ifndef _CSERIALIZABLELUA_H
#define _CSERIALIZABLELUA_H

#ifdef SERIALIZABLE_USE_LUA
extern "C"
{
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}
#else
class lua_State;
#endif

#include <string>

namespace codeframe
{

    /*****************************************************************************/
    /**
      * @brief Scripting functionality
      * @author Sebastian Milosz
      * @version 1.0
      * @note cSerializableLua
     **
    ******************************************************************************/
    class cSerializableLua
    {
        private:
            lua_State* m_luastate;

        protected:
            void        ThisToLua( lua_State* l, bool classDeclaration = true );

        public:
                     cSerializableLua( cSerializableInterface& sint );
            virtual ~cSerializableLua();

            void LuaRunString( std::string luaScriptString );
            void LuaRunFile  ( std::string luaScriptFile   );
    };

}

#endif
