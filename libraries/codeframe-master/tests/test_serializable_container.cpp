#include "catch.hpp"

#include "test_serializable_fixture.hpp"

TEST_CASE( "codeframe library Object Container", "[codeframe][ObjectContainer]" )
{
    smart_ptr<ObjectNode> staticSerializableObject( new classTest_Static("testNameStatic", nullptr) );
    smart_ptr<ObjectNode> staticContainerObject( new classTest_Container("testNameContainerStatic", staticSerializableObject) );

    SECTION( "test Create/Dispose ByBuildType" )
    {
        REQUIRE( staticContainerObject->Path().IsNameUnique("node[0]") == true );
        REQUIRE( staticContainerObject->Path().IsNameUnique("node[1]") == true );
        REQUIRE( staticContainerObject->Path().IsNameUnique("node[2]") == true );
        REQUIRE( staticContainerObject->Path().IsNameUnique("node[3]") == true );
        REQUIRE( staticContainerObject->Path().IsNameUnique("node[4]") == true );

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

        REQUIRE( staticContainerObject->Path().IsNameUnique("node[0]") == false );
        REQUIRE( staticContainerObject->Path().IsNameUnique("node[1]") == false );
        REQUIRE( staticContainerObject->Path().IsNameUnique("node[2]") == false );
        REQUIRE( staticContainerObject->Path().IsNameUnique("node[3]") == false );
        REQUIRE( staticContainerObject->Path().IsNameUnique("node[4]") == false );

        smart_ptr<ObjectContainer> containerObject = smart_dynamic_pointer_cast<ObjectContainer>(staticContainerObject);

        REQUIRE( smart_ptr_isValid(containerObject) );

        REQUIRE( containerObject->DisposeByBuildType(codeframe::DYNAMIC) );

        REQUIRE( staticContainerObject->Path().IsNameUnique("node[0]") == true );
        REQUIRE( staticContainerObject->Path().IsNameUnique("node[1]") == true );
        REQUIRE( staticContainerObject->Path().IsNameUnique("node[2]") == true );
        REQUIRE( staticContainerObject->Path().IsNameUnique("node[3]") == true );
        REQUIRE( staticContainerObject->Path().IsNameUnique("node[4]") == true );
    }
}
