#include "catch.hpp"

#include "test_serializable_fixture.hpp"

TEST_CASE( "codeframe library Object Create/Delete", "[codeframe][Create_Delete]" )
{
    SECTION( "test normal Create/Dispose ByBuildType" )
    {
        classTest_Container* objPtr1 = new classTest_Container("testNameContainerStatic1", nullptr);
        classTestStaticDynamic_Container* objPtr2 = new classTestStaticDynamic_Container("testNameContainerStatic2", nullptr);
        classTest_SubInternal* objPtr3 = new  classTest_SubInternal("testNameContainerStatic3", nullptr);
        classTest_Dynamic* objPtr4 = new classTest_Dynamic("testNameContainerStatic4", nullptr);

        delete objPtr4;
        delete objPtr3;
        delete objPtr2;
        delete objPtr1;
    }

    SECTION( "test smart Create/Dispose ByBuildType" )
    {
        classTest_Container* objPtr1 = new classTest_Container("testNameContainerStatic2", nullptr);
        classTestStaticDynamic_Container* objPtr2 = new classTestStaticDynamic_Container("testNameContainerStatic1", nullptr);
        classTest_Dynamic* objPtr3 = new classTest_Dynamic("testNameContainerStatic3", nullptr);

        smart_ptr<ObjectNode> staticContainerObject1( objPtr1 );
        staticContainerObject1 = nullptr;

        smart_ptr<ObjectNode> staticContainerObject2( objPtr2 );
        staticContainerObject2 = nullptr;

        smart_ptr<ObjectNode> staticContainerObject3( objPtr3 );
        staticContainerObject3 = nullptr;
    }
}
