#include "catch.hpp"

#include "test_serializable_fixture.hpp"

TEST_CASE( "codeframe library object path", "[Object::Path]" )
{
    smart_ptr<ObjectNode> staticSerializableObject( new classTest_Static("testNameStatic", nullptr) );
    smart_ptr<ObjectNode> staticContainerObject( new classTest_Container("testNameContainerStatic", staticSerializableObject) );

    smart_ptr<ObjectNode> node0 = staticContainerObject->Create( "classTest_Dynamic", "node" );    // node[0]
    smart_ptr<ObjectNode> node1 = staticContainerObject->Create( "classTest_Dynamic", "node" );    // node[1]
    smart_ptr<ObjectNode> node2 = staticContainerObject->Create( "classTest_Dynamic", "node" );    // node[2]
    smart_ptr<ObjectNode> node3 = staticContainerObject->Create( "classTest_Dynamic", "node" );    // node[3]
    smart_ptr<ObjectNode> node4 = staticContainerObject->Create( "classTest_Dynamic", "node" );    // node[4]

    SECTION( "Basic codeframe library objects tests" )
    {
        REQUIRE( staticSerializableObject->ObjectName() == "testNameStatic" );
        REQUIRE( staticSerializableObject->Child("testNameContainerStatic")->ObjectName() == "testNameContainerStatic" );

        REQUIRE( staticContainerObject->Count() == 5 );

        REQUIRE( staticContainerObject->Child(0)->Property("Property1")->GetValue<int>() == 100 );

        // Test object PathString
        REQUIRE( node3->Path().PathString() == "testNameStatic/testNameContainerStatic/node[3]" );
        REQUIRE( node4->Path().PathString() == "testNameStatic/testNameContainerStatic/node[4]" );

        // Direct property access
        smart_ptr<PropertyNode> propNode = staticSerializableObject->PropertyFromPath( "testNameStatic/testNameContainerStatic/node[0].Property1" );
        REQUIRE( smart_ptr_isValid( propNode ) );
        REQUIRE( propNode->GetValue<int>() == 100 );

        *propNode = 101;

        REQUIRE( propNode->GetValue<int>() == 101 );

        // Script Test
        staticSerializableObject->Script().RunString("CF:GetProperty('testNameStatic/testNameContainerStatic/node[0].Property1').Number = 1");
        staticSerializableObject->Script().RunString("CF:GetProperty('testNameStatic/testNameContainerStatic/node[0].Property2').Number = 789");

        REQUIRE( propNode->GetValue<int>() == 1 );

        propNode = staticSerializableObject->PropertyFromPath( "testNameStatic/testNameContainerStatic/node[0].Property2" );

        REQUIRE( propNode->GetValue<int>() == 789 );

        // Property access by selection
        propNode = staticSerializableObject->PropertyFromPath( "testNameStatic/testNameContainerStatic/node[*].Property1" );

        REQUIRE( smart_ptr_isValid( propNode ) );

        INFO ( "The selection property name: " << propNode->Name() );
        INFO ( "The selection parent name: " << propNode->ParentName() );

        *propNode = 777;

        REQUIRE( staticSerializableObject->PropertyFromPath( "testNameStatic/testNameContainerStatic/node[0].Property1" )->GetValue<int>() == 777 );
        REQUIRE( staticSerializableObject->PropertyFromPath( "testNameStatic/testNameContainerStatic/node[1].Property1" )->GetValue<int>() == 777 );
        REQUIRE( staticSerializableObject->PropertyFromPath( "testNameStatic/testNameContainerStatic/node[2].Property1" )->GetValue<int>() == 777 );
        REQUIRE( staticSerializableObject->PropertyFromPath( "testNameStatic/testNameContainerStatic/node[3].Property1" )->GetValue<int>() == 777 );
        REQUIRE( staticSerializableObject->PropertyFromPath( "testNameStatic/testNameContainerStatic/node[4].Property1" )->GetValue<int>() == 777 );

        staticSerializableObject->Script().RunString("CF:GetProperty('testNameStatic/testNameContainerStatic/node[*].Property1').Number = 1234");

        REQUIRE( staticSerializableObject->PropertyFromPath( "testNameStatic/testNameContainerStatic/node[0].Property1" )->GetValue<int>() == 1234 );
        REQUIRE( staticSerializableObject->PropertyFromPath( "testNameStatic/testNameContainerStatic/node[1].Property1" )->GetValue<int>() == 1234 );
        REQUIRE( staticSerializableObject->PropertyFromPath( "testNameStatic/testNameContainerStatic/node[2].Property1" )->GetValue<int>() == 1234 );
        REQUIRE( staticSerializableObject->PropertyFromPath( "testNameStatic/testNameContainerStatic/node[3].Property1" )->GetValue<int>() == 1234 );
        REQUIRE( staticSerializableObject->PropertyFromPath( "testNameStatic/testNameContainerStatic/node[4].Property1" )->GetValue<int>() == 1234 );
    }

    SECTION( "Test Relative Paths" )
    {
        smart_ptr<PropertyNode> propNode = staticSerializableObject->PropertyFromPath( "testNameStatic/testNameContainerStatic/node[3].Property1" );
        smart_ptr<PropertyNode> node3_1 = staticContainerObject->PropertyFromPath("/node[3].Property1" );
        smart_ptr<PropertyNode> node3_2 = node4->PropertyFromPath("/../../testNameContainerStatic/node[3].Property1" );

        *propNode = 1234;

        REQUIRE( smart_ptr_isValid( node3_1 ) );
        REQUIRE( node3_1->GetValue<int>() == 1234 );

        *propNode = 1235;

        REQUIRE( smart_ptr_isValid( node3_2 ) );
        REQUIRE( node3_2->GetValue<int>() == 1235 );
    }

    SECTION( "Test Property ReferencePath" )
    {
        staticContainerObject->Child(1)->Property("PropertyLink")->SetValue(3344U);
        REQUIRE( staticContainerObject->Child(0)->Property("Property1")->GetValue<int>() == 3344U );

        staticContainerObject->Child(1)->Property("PropertyLink_rel")->SetValue(6522U);
        REQUIRE( staticContainerObject->Child(0)->Property("Property1")->GetValue<int>() == 6522U );
    }
}
