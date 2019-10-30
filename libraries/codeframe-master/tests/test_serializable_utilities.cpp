#include "catch.hpp"

#include <string>
#include <map>

#include <utilities/DataTypesUtilities.h>

TEST_CASE( "Serializable library DataTypesUtilities.h : CircularBuffer", "[DataTypesUtilities:CircularBuffer]" )
{
    class TestDataStorage : public utilities::data::DataStorage
    {
        public:
           virtual void Add( const std::string& key, const std::string& value )
           {
                m_DataMap[ key ] = value;
           }

           virtual void Get( const std::string& key, std::string& value )
           {
                std::map<std::string, std::string>::iterator it = m_DataMap.find( key );
                if ( it != m_DataMap.end() )
                {
                    value = it->second;
                }
           }
        private:
            std::map<std::string, std::string>  m_DataMap;
    };

    TestDataStorage ds;

    utilities::data::CircularBuffer<8, std::string> g_CircularBuffer_w;
    utilities::data::CircularBuffer<8, std::string> g_CircularBuffer_r;

    REQUIRE( g_CircularBuffer_w.IsEmpty() == true );

    g_CircularBuffer_w.Push( "Test/String/1" );
    g_CircularBuffer_w.Push( "Test/String/2" );
    g_CircularBuffer_w.Push( "Test/String/3" );
    g_CircularBuffer_w.Push( "Test/String/4" );
    g_CircularBuffer_w.Push( "Test/String/5" );
    g_CircularBuffer_w.Push( "Test/String/6" );
    g_CircularBuffer_w.Push( "Test/String/7" );

    g_CircularBuffer_w.Save( ds );
    g_CircularBuffer_r.Load( ds );

    SECTION( "Test PeekPrew and PeekNext functionality before load" )
    {
        REQUIRE( g_CircularBuffer_w.PeekPrew() == "Test/String/7" );
        REQUIRE( g_CircularBuffer_w.PeekPrew() == "Test/String/6" );
        REQUIRE( g_CircularBuffer_w.PeekPrew() == "Test/String/5" );
        REQUIRE( g_CircularBuffer_w.PeekPrew() == "Test/String/4" );
        REQUIRE( g_CircularBuffer_w.PeekPrew() == "Test/String/3" );
        REQUIRE( g_CircularBuffer_w.PeekPrew() == "Test/String/2" );
        REQUIRE( g_CircularBuffer_w.PeekPrew() == "Test/String/1" );
        REQUIRE( g_CircularBuffer_w.PeekPrew() == "Test/String/7" );
        REQUIRE( g_CircularBuffer_w.PeekPrew() == "Test/String/6" );
        REQUIRE( g_CircularBuffer_w.PeekNext() == "Test/String/7" );
        REQUIRE( g_CircularBuffer_w.PeekNext() == "Test/String/1" );
        REQUIRE( g_CircularBuffer_w.PeekPrew() == "Test/String/7" );
    }

    SECTION( "Test PeekPrew and PeekNext functionality before load" )
    {
        REQUIRE( g_CircularBuffer_r.PeekPrew() == "Test/String/7" );
        REQUIRE( g_CircularBuffer_r.PeekPrew() == "Test/String/6" );
        REQUIRE( g_CircularBuffer_r.PeekPrew() == "Test/String/5" );
        REQUIRE( g_CircularBuffer_r.PeekPrew() == "Test/String/4" );
        REQUIRE( g_CircularBuffer_r.PeekPrew() == "Test/String/3" );
        REQUIRE( g_CircularBuffer_r.PeekPrew() == "Test/String/2" );
        REQUIRE( g_CircularBuffer_r.PeekPrew() == "Test/String/1" );
        REQUIRE( g_CircularBuffer_r.PeekPrew() == "Test/String/7" );
        REQUIRE( g_CircularBuffer_r.PeekPrew() == "Test/String/6" );
        REQUIRE( g_CircularBuffer_r.PeekNext() == "Test/String/7" );
        REQUIRE( g_CircularBuffer_r.PeekNext() == "Test/String/1" );
        REQUIRE( g_CircularBuffer_r.PeekPrew() == "Test/String/7" );
    }
}
