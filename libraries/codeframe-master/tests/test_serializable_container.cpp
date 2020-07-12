#include "catch.hpp"

#include "test_serializable_fixture.hpp"

TEST_CASE( "codeframe library Object Container construction and destruction", "[codeframe::Object]" )
{
    smart_ptr<ObjectNode> staticSerializableObject( new classTest_Static("testNameStatic", nullptr) );
    smart_ptr<ObjectNode> staticContainerObject( new classTest_Container("testNameContainerStatic", staticSerializableObject) );

    SECTION( "test Create/Dispose" )
    {
        smart_ptr<ObjectSelection> node0 = staticContainerObject->Create( "classTest_Dynamic", "node" );    // node[0]
        smart_ptr<ObjectSelection> node1 = staticContainerObject->Create( "classTest_Dynamic", "node" );    // node[1]
        smart_ptr<ObjectSelection> node2 = staticContainerObject->Create( "classTest_Dynamic", "node" );    // node[2]
        smart_ptr<ObjectSelection> node3 = staticContainerObject->Create( "classTest_Dynamic", "node" );    // node[3]
        smart_ptr<ObjectSelection> node4 = staticContainerObject->Create( "classTest_Dynamic", "node" );    // node[4]

        REQUIRE( smart_ptr_isValid( node0 ) );
        REQUIRE( smart_ptr_isValid( node1 ) );
        REQUIRE( smart_ptr_isValid( node2 ) );
        REQUIRE( smart_ptr_isValid( node3 ) );
        REQUIRE( smart_ptr_isValid( node4 ) );

        REQUIRE( staticContainerObject->Count() == 5U);
    }
}
