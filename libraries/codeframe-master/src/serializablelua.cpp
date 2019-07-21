#include "serializable.hpp"

#ifdef SERIALIZABLE_USE_LUA
#include <LuaBridge/LuaBridge.h>
using namespace luabridge;
#endif
#include <fstream>          // std::ifstream

#include <LoggerUtilities.h>

namespace codeframe
{

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableScript::cSerializableScript( cSerializableInterface& sint ) :
        m_sint( sint ),
        m_luastate( NULL )
    {

    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cSerializableScript::~cSerializableScript()
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
    void cSerializableScript::ThisToLua( lua_State* l, bool classDeclaration )
    {
        #ifdef SERIALIZABLE_USE_LUA

        if( classDeclaration )
        {
            getGlobalNamespace( l )
            .beginNamespace( "CLASS" )
                .beginClass<PropertyBase>( "PropertyBase" )
                    .addProperty( "Number", &PropertyBase::GetNumber, &PropertyBase::SetNumber )
                    .addProperty( "String", &PropertyBase::GetString, &PropertyBase::SetString )
                    .addProperty( "Real"  , &PropertyBase::GetReal,   &PropertyBase::SetReal   )
                .endClass()
            .endNamespace();

            // Add accessors function
            //setGlobal( l, m_sint, "mc" );
        }

        // Po wszystkich propertisach dodajemy do lua
        for( PropertyIterator it = m_sint.PropertyManager().begin(); it != m_sint.PropertyManager().end(); ++it )
        {
            PropertyBase* iser = *it;

            std::string namespaceLUAName = iser->Path();
            std::string objectLUAName    = iser->Name();

            // Ze skryptu lua propertisy widoczne sa w przestrzeniach nazw odpowiadajacych
            // ich sciezce, mozliwa jest tylko i wylacznie zmiana wartosci dla ulatwienia sprawy
            //namespaceLUAName.c_str()

            setGlobal( l, iser, "mc" );
        }

        // Po wszystkich obiektach dzieci dodajemy do lua
        for ( cSerializableChildList::iterator it = m_sint.ChildList().begin(); it != m_sint.ChildList().end(); ++it )
        {
            cSerializableInterface* iser = *it;
            if ( iser )
            {
                iser->Script().ThisToLua( l, false );
            }
        }

        #else
        (void)l;
        (void)classDeclaration;
        #endif
    }

    /*****************************************************************************/
    /**
      * @brief
      * @param scriptString script to run
      * @param thread - if true script is executed in new thread
     **
    ******************************************************************************/
    void cSerializableScript::RunString( std::string scriptString )
    {
        #ifdef SERIALIZABLE_USE_LUA

        m_luastate = luaL_newstate();

        if( m_luastate == NULL ) return;

        try
        {
            luaL_openlibs( m_luastate );

            ThisToLua( m_luastate );

            if( luaL_loadstring( m_luastate, scriptString.c_str() ) != 0 )
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
        (void)scriptString;
        #endif
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cSerializableScript::RunFile( std::string scriptFile )
    {
        #ifdef SERIALIZABLE_USE_LUA

        std::ifstream t( scriptFile.c_str() );
        std::stringstream buffer;
        buffer << t.rdbuf();

        RunString( buffer.str() );

        #else
        (void)scriptFile;
        #endif
    }

}
