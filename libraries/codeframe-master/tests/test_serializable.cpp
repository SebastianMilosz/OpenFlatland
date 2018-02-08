#include "catch.hpp"

#include "serializable.h"

TEST_CASE( "Serializable library construction and destruction", "[serializable]" )
{
    class classTestSerializable : public codeframe::cSerializable
    {
        public:
        std::string Role()      { return "Object";                }
        std::string Class()     { return "classTestSerializable"; }
        std::string BuildType() { return "Static";                }

        public:
            classTestSerializable( std::string name, cSerializable* parent ) : cSerializable( name, parent )
            {

            }
    };

    classTestSerializable testSerializable( "testName", NULL );

    SECTION( "resizing bigger changes size and capacity" )
    {
        REQUIRE( testSerializable.ObjectName() == "testName" );
    }
}
