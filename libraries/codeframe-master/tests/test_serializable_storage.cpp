#include "catch.hpp"

#include "test_serializable_fixture.hpp"

TEST_CASE( "codeframe Object save and restore", "[codeframe][Object][Storage]" )
{
    const std::string apiDir( utilities::file::GetExecutablePath() );
    const std::string dataFilePath( apiDir + std::string("\\test_data.xml") );

    smart_ptr<ObjectNode> staticSerializableObject( new classTest_Static("testNameStatic", nullptr) );
    smart_ptr<ObjectNode> staticContainerObject( new classTest_Container("testNameContainerStatic", staticSerializableObject) );

    smart_ptr<ObjectSelection> node0 = staticContainerObject->Create( "classTest_Dynamic", "node" );    // node[0]
    smart_ptr<ObjectSelection> node1 = staticContainerObject->Create( "classTest_Dynamic", "node" );    // node[1]
    smart_ptr<ObjectSelection> node2 = staticContainerObject->Create( "classTest_Dynamic", "node" );    // node[2]
    smart_ptr<ObjectSelection> node3 = staticContainerObject->Create( "classTest_Dynamic", "node" );    // node[3]
    smart_ptr<ObjectSelection> node4 = staticContainerObject->Create( "classTest_Dynamic", "node" );    // node[4]

    SECTION( "test get object name" )
    {
        REQUIRE( staticSerializableObject->ObjectName() == "testNameStatic" );
        REQUIRE( staticContainerObject->ObjectName() == "testNameContainerStatic" );
    }

    SECTION( "test save/restore" )
    {
        node0->Property("Property1")->SetValue(3301U);
        node0->Property("Property2")->SetValue(3302U);
        node0->Property("Property3")->SetValue(3303U);
        node0->Property("Property4")->SetValue(3304U);

        node1->Property("Property1")->SetValue(3311U);
        node1->Property("Property2")->SetValue(3312U);
        node1->Property("Property3")->SetValue(3313U);
        node1->Property("Property4")->SetValue(3314U);

        node2->Property("Property1")->SetValue(3321U);
        node2->Property("Property2")->SetValue(3322U);
        node2->Property("Property3")->SetValue(3323U);
        node2->Property("Property4")->SetValue(3324U);

        node3->Property("Property1")->SetValue(3331U);
        node3->Property("Property2")->SetValue(3332U);
        node3->Property("Property3")->SetValue(3333U);
        node3->Property("Property4")->SetValue(3334U);

        node4->Property("Property1")->SetValue(3341U);
        node4->Property("Property2")->SetValue(3342U);
        node4->Property("Property3")->SetValue(3343U);
        node4->Property("Property4")->SetValue(3344U);

        REQUIRE( node0->Property("Property1")->GetValue<unsigned int>() == 3301U );

        REQUIRE_NOTHROW( staticSerializableObject->Storage().SaveToFile( dataFilePath ) );

        node0->Property("Property1")->SetValue(4401U);
        node0->Property("Property2")->SetValue(4402U);
        node0->Property("Property3")->SetValue(4403U);
        node0->Property("Property4")->SetValue(4404U);

        node1->Property("Property1")->SetValue(4411U);
        node1->Property("Property2")->SetValue(4412U);
        node1->Property("Property3")->SetValue(4413U);
        node1->Property("Property4")->SetValue(4414U);

        node2->Property("Property1")->SetValue(4421U);
        node2->Property("Property2")->SetValue(4422U);
        node2->Property("Property3")->SetValue(4423U);
        node2->Property("Property4")->SetValue(4424U);

        node3->Property("Property1")->SetValue(4431U);
        node3->Property("Property2")->SetValue(4432U);
        node3->Property("Property3")->SetValue(4433U);
        node3->Property("Property4")->SetValue(4434U);

        node4->Property("Property1")->SetValue(4441U);
        node4->Property("Property2")->SetValue(4442U);
        node4->Property("Property3")->SetValue(4443U);
        node4->Property("Property4")->SetValue(4444U);

        REQUIRE( node0->Property("Property1")->GetValue<unsigned int>() == 4401U );

        // Restore object state from a file
        REQUIRE_NOTHROW( staticSerializableObject->Storage().LoadFromFile( dataFilePath ) );

        // After load all dynamic object selection should not be valid
        REQUIRE( node0->IsValid() == false );
        REQUIRE( node1->IsValid() == false );
        REQUIRE( node2->IsValid() == false );
        REQUIRE( node3->IsValid() == false );
        REQUIRE( node4->IsValid() == false );

        // Update selections after load
        node0 = staticContainerObject->Child("node[0]");    // node[0]
        node1 = staticContainerObject->Child("node[1]");    // node[1]
        node2 = staticContainerObject->Child("node[2]");    // node[2]
        node3 = staticContainerObject->Child("node[3]");    // node[3]
        node4 = staticContainerObject->Child("node[4]");    // node[4]

        // Verify loaded values
        REQUIRE( node0->Property("Property1")->GetValue<unsigned int>() == 3301U );
        REQUIRE( node1->Property("Property1")->GetValue<unsigned int>() == 3311U );
        REQUIRE( node2->Property("Property1")->GetValue<unsigned int>() == 3321U );
        REQUIRE( node3->Property("Property1")->GetValue<unsigned int>() == 3331U );
        REQUIRE( node4->Property("Property1")->GetValue<unsigned int>() == 3341U );

        REQUIRE( ReferenceManager::UnresolvedReferencesCount() == 0U );
    }
}
