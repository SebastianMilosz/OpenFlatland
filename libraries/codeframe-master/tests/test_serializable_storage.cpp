#include "catch.hpp"

#include "test_serializable_fixture.hpp"

TEST_CASE( "codeframe library Object construction and destruction", "[codeframe::Object]" )
{
    const std::string apiDir( utilities::file::GetExecutablePath() );
    const std::string dataFilePath( apiDir + std::string("\\test_data.xml") );

    smart_ptr<ObjectNode> staticSerializableObject( new classTest_Static("testNameStatic", nullptr) );
    smart_ptr<ObjectNode> staticContainerObject( new classTest_Container("testNameContainerStatic", staticSerializableObject) );

    smart_ptr<ObjectNode> node0 = staticContainerObject->Create( "classTest_Dynamic", "node" );    // node[0]
    smart_ptr<ObjectNode> node1 = staticContainerObject->Create( "classTest_Dynamic", "node" );    // node[1]
    smart_ptr<ObjectNode> node2 = staticContainerObject->Create( "classTest_Dynamic", "node" );    // node[2]
    smart_ptr<ObjectNode> node3 = staticContainerObject->Create( "classTest_Dynamic", "node" );    // node[3]
    smart_ptr<ObjectNode> node4 = staticContainerObject->Create( "classTest_Dynamic", "node" );    // node[4]

    SECTION( "test get object name" )
    {
        REQUIRE( staticSerializableObject->ObjectName() == "testNameStatic" );
        REQUIRE( staticContainerObject->ObjectName() == "testNameContainerStatic" );
    }

    SECTION( "test save/restore" )
    {
        staticSerializableObject->Storage().SaveToFile( dataFilePath );
    }
}
