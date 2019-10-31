#include "catch.hpp"

#include <serializable.hpp>

TEST_CASE( "Serializable library path", "[serializable.Path]" )
{
    class classTestSerializable : public codeframe::cSerializable
    {
        public:
            CODEFRAME_META_CLASS_NAME( "classTestSerializable" );
            CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

        public:
            classTestSerializable( const std::string& name, cSerializableInterface* parent ) : cSerializable( name, parent )
            {

            }
    };

    classTestSerializable testSerializable( "testName", NULL );

    SECTION( "resizing bigger changes size and capacity" )
    {
        REQUIRE( testSerializable.Identity().ObjectName() == "testName" );
    }
}
