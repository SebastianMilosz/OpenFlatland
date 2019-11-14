#include "catch.hpp"

#include <serializable_object.hpp>

TEST_CASE( "codeframe library Object construction and destruction", "[codeframe::Object]" )
{
    class classTestSerializable : public codeframe::Object
    {
        public:
            CODEFRAME_META_CLASS_NAME( "classTestObject" );
            CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

        public:
            classTestSerializable( const std::string& name, ObjectNode* parent ) : Object( name, parent )
            {

            }
    };

    classTestSerializable testSerializable( "testName", NULL );

    SECTION( "resizing bigger changes size and capacity" )
    {
        REQUIRE( testSerializable.Identity().ObjectName() == "testName" );
    }
}
