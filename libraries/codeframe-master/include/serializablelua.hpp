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
    class cSerializableScript
    {
        private:
            lua_State* m_luastate;
            cSerializableInterface& m_sint;

        protected:
            void ThisToLua( lua_State* l, bool classDeclaration = true );

            PropertyBase& GetProperty( std::string path );

        public:
                     cSerializableScript( cSerializableInterface& sint );
            virtual ~cSerializableScript();

            void RunString( std::string scriptString );
            void RunFile  ( std::string scriptFile   );
    };

}

#endif
