#include "serializable.h"

#ifdef SERIALIZABLE_USE_LUA
#include <LuaBridge.h>
using namespace luabridge;
#endif
#include <fstream>          // std::ifstream

namespace codeframe
{

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableLua::cSerializableLua() : m_luastate( NULL )
    {

    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableLua::~cSerializableLua()
    {
        #ifdef SERIALIZABLE_USE_LUA
        if( m_luastate ) { lua_close( m_luastate ); }
        #endif
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cSerializableLua::ThisToLua( lua_State* l, bool classDeclaration )
    {
        #ifdef SERIALIZABLE_USE_LUA

        if( classDeclaration )
        {
            getGlobalNamespace( l )
            .beginNamespace( "CLASS" )
                .beginClass<Property>( "Property" )
                    .addProperty( "Number", &Property::GetNumber, &Property::SetNumber )
                    .addProperty( "String", &Property::GetString, &Property::SetString )
                    .addProperty( "Real"  , &Property::GetReal,   &Property::SetReal   )
                .endClass()
            .endNamespace();
        }

        // Po wszystkich propertisach dodajemy do lua
        for( iterator it = this->begin(); it != this->end(); ++it )
        {
            Property* iser = *it;

            std::string namespaceLUAName = iser->Path();
            std::string objectLUAName    = iser->Name();

            // Ze skryptu lua propertisy widoczne sa w przestrzeniach nazw odpowiadajacych
            // ich sciezce, mozliwa jest tylko i wylacznie zmiana wartosci dla ulatwienia sprawy
            push( l, iser );
            lua_setglobal( l, namespaceLUAName.c_str() );

            /*
            getGlobalNamespace( l )
            .beginNamespace( namespaceLUAName.c_str() )
                .addVariable( objectLUAName.c_str() , (Property*)iser)
            .endNamespace();
            */
        }

        // Po wszystkich obiektach dzieci dodajemy do lua
        for( cChildList::iterator it = this->ChildList()->begin(); it != this->ChildList()->end(); ++it )
        {
            cSerializableLua* iser = *it;
            if(iser) { iser->ThisToLua( l, false ); }
        }

        #else
        (void)l;
        (void)classDeclaration;
        #endif
    }

    /*****************************************************************************/
    /**
      * @brief
      * @param s - lua script to run
      * @param thread - if true script is executed in new thread
     **
    ******************************************************************************/
    void cSerializableLua::LuaRunString( std::string luaScriptString )
    {
        #ifdef SERIALIZABLE_USE_LUA

        m_luastate = luaL_newstate();

        if( m_luastate == NULL ) return;

        try
        {
            luaL_openlibs( m_luastate );

            ThisToLua( m_luastate );

            if( luaL_loadstring( m_luastate, luaScriptString.c_str() ) != 0 )
            {
                // compile-time error
                LOGGER( LOG_ERROR << "LUA script compile-time error: " << lua_tostring( m_luastate, -1 ) );
                lua_close( m_luastate );
                m_luastate = NULL;
            }
            else if( lua_pcall( m_luastate, 0, 0, 0 ) != 0 )
            {
                // runtime error
                LOGGER( LOG_ERROR << "LUA script runtime error: " << lua_tostring( m_luastate, -1 ) );
                lua_close( m_luastate );
                m_luastate = NULL;
            }
        }
        catch(const std::runtime_error& re)
        {
            LOGGER( LOG_ERROR << "LUA script runtime exception: " << re.what() );
        }
        catch(...)
        {
            LOGGER( LOG_ERROR << "LUA script Critical Unknown failure occured. Possible memory corruption" );
        }

        #else
        (void)luaScriptString;
        #endif
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cSerializableLua::LuaRunFile( std::string luaScriptFile )
    {
        #ifdef SERIALIZABLE_USE_LUA

        std::ifstream t( luaScriptFile.c_str() );
        std::stringstream buffer;
        buffer << t.rdbuf();

        LuaRunString( buffer.str() );

        #else
        (void)luaScriptFile;
        #endif
    }

}
