#include "catch.hpp"

#include <serializable_object.hpp>
#include <serializable_object_container.hpp>

using namespace codeframe;

TEST_CASE( "codeframe library object path", "[Object::Path]" )
{
    class classTest_Static : public codeframe::Object
    {
        public:
            CODEFRAME_META_CLASS_NAME( "classTest_Static" );
            CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

        public:
            classTest_Static( const std::string& name, ObjectNode* parent ) : Object( name, parent )
            {
            }
    };

    class classTest_Dynamic : public codeframe::Object
    {
        public:
            CODEFRAME_META_CLASS_NAME( "classTest_Dynamic" );
            CODEFRAME_META_BUILD_TYPE( codeframe::DYNAMIC );

        codeframe::Property<int> Property1;
        codeframe::Property<int> Property2;
        codeframe::Property<int> Property3;
        codeframe::Property<int> Property4;

        codeframe::Property<int> PropertyLink;

        public:
            classTest_Dynamic( const std::string& name, ObjectNode* parent ) :
                Object( name, parent ),
                Property1( this, "Property1", 100U , cPropertyInfo().Kind( KIND_NUMBER ).Description("Property1_desc") ),
                Property2( this, "Property2", 200U , cPropertyInfo().Kind( KIND_NUMBER ).Description("Property2_desc") ),
                Property3( this, "Property3", 300U , cPropertyInfo().Kind( KIND_NUMBER ).Description("Property3_desc") ),
                Property4( this, "Property4", 400U , cPropertyInfo().Kind( KIND_NUMBER ).Description("Property4_desc") ),

                PropertyLink( this, "PropertyLink", 500U,
                                cPropertyInfo().
                                    Kind( KIND_NUMBER ).
                                    ReferencePath("testNameStatic/testNameContainerStatic/node[0].Property1").
                                    Description("Property4_desc") )
            {
            }
    };

    class classTest_Container : public codeframe::ObjectContainer
    {
        public:
            CODEFRAME_META_CLASS_NAME( "classTest_Container" );
            CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

            virtual smart_ptr<ObjectNode> Create(
                                                             const std::string& className,
                                                             const std::string& objName,
                                                             const std::vector<codeframe::VariantValue>& params = std::vector<codeframe::VariantValue>()
                                                             )
            {
                if ( className == "classTest_Dynamic" )
                {
                    smart_ptr<classTest_Dynamic> obj = smart_ptr<classTest_Dynamic>( new classTest_Dynamic( objName, this ) );

                    int id = InsertObject( obj );

                    return obj;
                }

                return smart_ptr<codeframe::ObjectNode>();
            }

        public:
            classTest_Container( const std::string& name, ObjectNode* parent ) : ObjectContainer( name, parent )
            {
            }
    };

    classTest_Static staticSerializableObject( "testNameStatic", NULL );

    classTest_Container staticContainerObject( "testNameContainerStatic", &staticSerializableObject );

    smart_ptr<ObjectNode> node0 = staticContainerObject.Create( "classTest_Dynamic", "node" );    // node[0]
    smart_ptr<ObjectNode> node1 = staticContainerObject.Create( "classTest_Dynamic", "node" );    // node[1]
    smart_ptr<ObjectNode> node2 = staticContainerObject.Create( "classTest_Dynamic", "node" );    // node[2]
    smart_ptr<ObjectNode> node3 = staticContainerObject.Create( "classTest_Dynamic", "node" );    // node[3]
    smart_ptr<ObjectNode> node4 = staticContainerObject.Create( "classTest_Dynamic", "node" );    // node[4]

    SECTION( "Basic codeframe library objects tests" )
    {
        REQUIRE( staticSerializableObject.Identity().ObjectName() == "testNameStatic" );
        REQUIRE( staticContainerObject.Identity().ObjectName() == "testNameContainerStatic" );

        REQUIRE( staticContainerObject.Count() == 5 );

        REQUIRE( staticContainerObject[0]->Property("Property1")->GetValue<int>() == 100 );

        // Test object PathString
        REQUIRE( node3->Path().PathString() == "testNameStatic/testNameContainerStatic/node[3]" );
        REQUIRE( node4->Path().PathString() == "testNameStatic/testNameContainerStatic/node[4]" );

        // Direct property access
        smart_ptr<PropertyNode> propNode = staticSerializableObject.PropertyList().GetPropertyFromPath( "testNameStatic/testNameContainerStatic/node[0].Property1" );
        REQUIRE( smart_ptr_isValid( propNode ) );
        REQUIRE( propNode->GetValue<int>() == 100 );

        *propNode = 101;

        REQUIRE( propNode->GetValue<int>() == 101 );

        // Script Test
        staticSerializableObject.Script().RunString("CF:GetProperty('testNameStatic/testNameContainerStatic/node[0].Property1').Number = 1");
        staticSerializableObject.Script().RunString("CF:GetProperty('testNameStatic/testNameContainerStatic/node[0].Property2').Number = 789");

        REQUIRE( propNode->GetValue<int>() == 1 );

        propNode = staticSerializableObject.PropertyList().GetPropertyFromPath( "testNameStatic/testNameContainerStatic/node[0].Property2" );

        REQUIRE( propNode->GetValue<int>() == 789 );

        // Property access by selection
        propNode = staticSerializableObject.PropertyList().GetPropertyFromPath( "testNameStatic/testNameContainerStatic/node[*].Property1" );

        REQUIRE( smart_ptr_isValid( propNode ) );

        INFO ( "The selection property name: " << propNode->Name() );
        INFO ( "The selection parent name: " << propNode->ParentName() );

        *propNode = 777;

        REQUIRE( staticSerializableObject.PropertyList().GetPropertyFromPath( "testNameStatic/testNameContainerStatic/node[0].Property1" )->GetValue<int>() == 777 );
        REQUIRE( staticSerializableObject.PropertyList().GetPropertyFromPath( "testNameStatic/testNameContainerStatic/node[1].Property1" )->GetValue<int>() == 777 );
        REQUIRE( staticSerializableObject.PropertyList().GetPropertyFromPath( "testNameStatic/testNameContainerStatic/node[2].Property1" )->GetValue<int>() == 777 );
        REQUIRE( staticSerializableObject.PropertyList().GetPropertyFromPath( "testNameStatic/testNameContainerStatic/node[3].Property1" )->GetValue<int>() == 777 );
        REQUIRE( staticSerializableObject.PropertyList().GetPropertyFromPath( "testNameStatic/testNameContainerStatic/node[4].Property1" )->GetValue<int>() == 777 );

        staticSerializableObject.Script().RunString("CF:GetProperty('testNameStatic/testNameContainerStatic/node[*].Property1').Number = 1234");

        REQUIRE( staticSerializableObject.PropertyList().GetPropertyFromPath( "testNameStatic/testNameContainerStatic/node[0].Property1" )->GetValue<int>() == 1234 );
        REQUIRE( staticSerializableObject.PropertyList().GetPropertyFromPath( "testNameStatic/testNameContainerStatic/node[1].Property1" )->GetValue<int>() == 1234 );
        REQUIRE( staticSerializableObject.PropertyList().GetPropertyFromPath( "testNameStatic/testNameContainerStatic/node[2].Property1" )->GetValue<int>() == 1234 );
        REQUIRE( staticSerializableObject.PropertyList().GetPropertyFromPath( "testNameStatic/testNameContainerStatic/node[3].Property1" )->GetValue<int>() == 1234 );
        REQUIRE( staticSerializableObject.PropertyList().GetPropertyFromPath( "testNameStatic/testNameContainerStatic/node[4].Property1" )->GetValue<int>() == 1234 );
    }

    SECTION( "Test Relative Paths" )
    {
        smart_ptr<PropertyNode> propNode = staticSerializableObject.PropertyList().GetPropertyFromPath( "testNameStatic/testNameContainerStatic/node[3].Property1" );
        smart_ptr<PropertyNode> node3_1 = staticContainerObject.PropertyFromPath("/node[3].Property1" );
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
        //staticContainerObject[1]->Property("PropertyLink")->SetValue(3344U);

        //REQUIRE( staticContainerObject[0]->Property("Property1")->GetValue<int>() == 3344U );
    }
}
