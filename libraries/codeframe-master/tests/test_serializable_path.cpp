#include "catch.hpp"

#include "test_serializable_fixture.hpp"

TEST_CASE( "codeframe library object path", "[codeframe][Object][Path]" )
{
    smart_ptr<ObjectNode> staticSerializableObject( new classTest_Static("testNameStatic", nullptr) );
    smart_ptr<ObjectNode> staticContainerObject( new classTest_Container("testNameContainerStatic", staticSerializableObject) );

    smart_ptr<codeframe::Object> node0 = staticContainerObject->Create("classTest_Dynamic"    , "node"); // node[0]
    smart_ptr<codeframe::Object> node1 = staticContainerObject->Create("classTest_Dynamic"    , "node"); // node[1]
    smart_ptr<codeframe::Object> node2 = staticContainerObject->Create("classTest_Dynamic"    , "node"); // node[2]
    smart_ptr<codeframe::Object> node3 = staticContainerObject->Create("classTest_Dynamic"    , "node"); // node[3]
    smart_ptr<codeframe::Object> node4 = staticContainerObject->Create("classTest_Dynamic"    , "node"); // node[4]
    smart_ptr<codeframe::Object> node5 = staticContainerObject->Create("classTest_Dynamic_rel", "node"); // node[5]

    SECTION( "Basic codeframe library objects tests" )
    {
        REQUIRE( staticSerializableObject->ObjectName() == "testNameStatic" );
        REQUIRE( staticSerializableObject->Child("testNameContainerStatic")->ObjectName() == "testNameContainerStatic" );

        REQUIRE( staticContainerObject->Count() == 6 );

        REQUIRE( staticContainerObject->Child(0)->Property("Property1")->GetValue<int>() == 100 );

        // Test object PathString
        REQUIRE( node3->Path().PathString() == "testNameStatic/testNameContainerStatic/node[3]" );
        REQUIRE( node4->Path().PathString() == "testNameStatic/testNameContainerStatic/node[4]" );

        // First check some improper path strings
        smart_ptr<PropertyNode> propNodeInvalid = staticSerializableObject->PropertyFromPath( "" );
        REQUIRE( smart_ptr_isValid( propNodeInvalid ) == false );

        // Property access by path
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

    SECTION( "Test Property Reverse ReferencePath" )
    {
        staticContainerObject->Child(4)->Property("Property_rew")->SetValue(5544U);
        REQUIRE( staticContainerObject->Child(5)->Property("PropertyLink_rel_rew")->GetValue<int>() == 5544U );
    }

    SECTION( "Test Script multiple selection" )
    {
        staticSerializableObject->Script().RunString("CF:GetProperty('testNameStatic/testNameContainerStatic/node[*].Property1').Number = 67892");

        REQUIRE( staticContainerObject->Child(0)->Property("Property1")->GetValue<int>() == 67892U );
        REQUIRE( staticContainerObject->Child(1)->Property("Property1")->GetValue<int>() == 67892U );
        REQUIRE( staticContainerObject->Child(2)->Property("Property1")->GetValue<int>() == 67892U );
        REQUIRE( staticContainerObject->Child(3)->Property("Property1")->GetValue<int>() == 67892U );
        REQUIRE( staticContainerObject->Child(4)->Property("Property1")->GetValue<int>() == 67892U );

        staticSerializableObject->Script().RunString("CF:GetProperty('testNameStatic/testNameContainerStatic/node[*].Property_float').Number = 5674");

        REQUIRE( staticContainerObject->Child(0)->Property("Property_float")->GetValue<float>() == 5674U );
        REQUIRE( staticContainerObject->Child(1)->Property("Property_float")->GetValue<float>() == 5674U );
        REQUIRE( staticContainerObject->Child(2)->Property("Property_float")->GetValue<float>() == 5674U );
        REQUIRE( staticContainerObject->Child(3)->Property("Property_float")->GetValue<float>() == 5674U );
        REQUIRE( staticContainerObject->Child(4)->Property("Property_float")->GetValue<float>() == 5674U );

        staticContainerObject->Script().RunString("CF:GetProperty('testNameStatic/testNameContainerStatic/node[*].Property_float').Number = 88888");

        REQUIRE( staticContainerObject->Child(0)->Property("Property_float")->GetValue<float>() == 88888U );
        REQUIRE( staticContainerObject->Child(1)->Property("Property_float")->GetValue<float>() == 88888U );
        REQUIRE( staticContainerObject->Child(2)->Property("Property_float")->GetValue<float>() == 88888U );
        REQUIRE( staticContainerObject->Child(3)->Property("Property_float")->GetValue<float>() == 88888U );
        REQUIRE( staticContainerObject->Child(4)->Property("Property_float")->GetValue<float>() == 88888U );

        staticContainerObject->Script().RunString("CF:GetProperty('testNameStatic/testNameContainerStatic/node[*]/InternalObject.Property_float').Number = 123456");

        REQUIRE( staticContainerObject->Child(0)->GetObjectByName("InternalObject")->Property("Property_float")->GetValue<float>() == 123456U );
        REQUIRE( staticContainerObject->Child(1)->GetObjectByName("InternalObject")->Property("Property_float")->GetValue<float>() == 123456U );
        REQUIRE( staticContainerObject->Child(2)->GetObjectByName("InternalObject")->Property("Property_float")->GetValue<float>() == 123456U );
        REQUIRE( staticContainerObject->Child(3)->GetObjectByName("InternalObject")->Property("Property_float")->GetValue<float>() == 123456U );
        REQUIRE( staticContainerObject->Child(4)->GetObjectByName("InternalObject")->Property("Property_float")->GetValue<float>() == 123456U );
    }
}
