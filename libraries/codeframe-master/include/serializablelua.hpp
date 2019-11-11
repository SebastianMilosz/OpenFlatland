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
      * @note cSerializableScript
     **
    ******************************************************************************/
    class cSerializableScript
    {
        public:
                     cSerializableScript( cSerializableInterface& sint );
            virtual ~cSerializableScript();

            void RunString( const std::string& scriptString );
            void RunFile  ( const std::string& scriptFile   );

        protected:
            void ThisToLua( lua_State* l );

        private:
            smart_ptr<PropertyNode> GetProperty( const std::string& path );

            lua_State* m_luastate;
            cSerializableInterface& m_sint;
    };

}

#endif
