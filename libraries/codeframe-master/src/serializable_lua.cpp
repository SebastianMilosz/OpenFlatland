#include "serializable_object.hpp"

#include "serializable_property_selection.hpp"

#ifdef SERIALIZABLE_USE_LUA
#include <LuaBridge/LuaBridge.h>
using namespace luabridge;

namespace luabridge
{
    template <class T>
    struct ContainerTraits < smart_ptr <T> >
    {
        typedef T Type;
        static T* get (smart_ptr <T> const& c)
        {
            return smart_ptr_getRaw( c );
        }
    };
}

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
    cScript::cScript( ObjectNode& sint ) :
        m_luastate( nullptr ),
        m_sint( sint )
    {

    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cScript::~cScript()
    {
        #ifdef SERIALIZABLE_USE_LUA
        if ( m_luastate )
        {
            lua_close( m_luastate );
        }
        #endif
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cScript::ThisToLua( lua_State* l )
    {
        #ifdef SERIALIZABLE_USE_LUA

        getGlobalNamespace( l )
        .beginNamespace( "CLASS" )
            .beginClass<PropertyNode>( "Property" )
                .addProperty( "Number", &PropertyNode::GetNumber, &PropertyNode::SetNumber )
                .addProperty( "String", &PropertyNode::GetString, &PropertyNode::SetString )
                .addProperty( "Real"  , &PropertyNode::GetReal,   &PropertyNode::SetReal   )
            .endClass()
            .deriveClass <PropertyBase, PropertyNode> ("PropertyBase")
                .addProperty( "Number", &PropertyBase::GetNumber, &PropertyBase::SetNumber )
                .addProperty( "String", &PropertyBase::GetString, &PropertyBase::SetString )
                .addProperty( "Real"  , &PropertyBase::GetReal,   &PropertyBase::SetReal   )
            .endClass ()
            .deriveClass <PropertySelection, PropertyNode> ("PropertySelection")
                .addProperty( "Number", &PropertySelection::GetNumber, &PropertySelection::SetNumber )
                .addProperty( "String", &PropertySelection::GetString, &PropertySelection::SetString )
                .addProperty( "Real"  , &PropertySelection::GetReal,   &PropertySelection::SetReal   )
            .endClass ()
            .beginClass<cScript>( "Script" )
                .addFunction("GetProperty", &cScript::GetProperty)
            .endClass()
        .endNamespace();

        // Add CodeFrame object to lua global scope
        setGlobal( l, this, "CF" );

        #else
        (void)l;
        (void)classDeclaration;
        #endif
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    smart_ptr<PropertyNode> cScript::GetProperty( const std::string& path )
    {
        return m_sint.PropertyList().GetPropertyFromPath( path );
    }

    /*****************************************************************************/
    /**
      * @brief
      * @param scriptString script to run
      * @param thread - if true script is executed in new thread
     **
    ******************************************************************************/
    void cScript::RunString( const std::string& scriptString )
    {
        #ifdef SERIALIZABLE_USE_LUA

        m_luastate = luaL_newstate();

        if ( m_luastate == nullptr )
        {
            return;
        }

        try
        {
            luaL_openlibs( m_luastate );

            ThisToLua( m_luastate );

            if ( luaL_loadstring( m_luastate, scriptString.c_str() ) != 0 )
            {
                // compile-time error
                LOGGER( LOG_ERROR << "LUA script compile-time error: " << lua_tostring( m_luastate, -1 ) );
                lua_close( m_luastate );
                m_luastate = nullptr;
            }
            else if ( lua_pcall( m_luastate, 0, 0, 0 ) != 0 )
            {
                // runtime error
                LOGGER( LOG_ERROR << "LUA script runtime error: " << lua_tostring( m_luastate, -1 ) );
                lua_close( m_luastate );
                m_luastate = nullptr;
            }
        }
        catch (const std::runtime_error& re)
        {
            LOGGER( LOG_ERROR << "LUA script runtime exception: " << re.what() );
        }
        catch (...)
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
    void cScript::RunFile( const std::string& scriptFile )
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
