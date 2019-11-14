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

        public:
            classTest_Dynamic( const std::string& name, ObjectNode* parent ) :
                Object( name, parent ),
                Property1( this, "Property1", 100U , cPropertyInfo().Kind( KIND_NUMBER ).Description("Property1_desc") ),
                Property2( this, "Property2", 200U , cPropertyInfo().Kind( KIND_NUMBER ).Description("Property2_desc") ),
                Property3( this, "Property3", 300U , cPropertyInfo().Kind( KIND_NUMBER ).Description("Property3_desc") ),
                Property4( this, "Property4", 400U , cPropertyInfo().Kind( KIND_NUMBER ).Description("Property4_desc") )
            {
            }
    };

    class classTest_Container : public codeframe::cSerializableContainer
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
            classTest_Container( const std::string& name, ObjectNode* parent ) : cSerializableContainer( name, parent )
            {
            }
    };

    classTest_Static staticSerializableObject( "testNameStatic", NULL );

    classTest_Container staticContainerObject( "testNameContainerStatic", &staticSerializableObject );

    staticContainerObject.Create( "classTest_Dynamic", "node" );
    staticContainerObject.Create( "classTest_Dynamic", "node" );
    staticContainerObject.Create( "classTest_Dynamic", "node" );
    staticContainerObject.Create( "classTest_Dynamic", "node" );
    staticContainerObject.Create( "classTest_Dynamic", "node" );

    SECTION( "Basic codeframe library objects tests" )
    {
        REQUIRE( staticSerializableObject.Identity().ObjectName() == "testNameStatic" );
        REQUIRE( staticContainerObject.Identity().ObjectName() == "testNameContainerStatic" );

        REQUIRE( staticContainerObject.Count() == 5 );

        REQUIRE( (int)(*static_cast<classTest_Dynamic*>(smart_ptr_getRaw(staticContainerObject[0]))).Property1 == 100 );

        // Direct property access
        smart_ptr<PropertyNode> propNode = staticSerializableObject.PropertyManager().GetPropertyFromPath( "testNameStatic/testNameContainerStatic/node[0].Property1" );
        REQUIRE( smart_ptr_isValid( propNode ) );
        REQUIRE( (int)(*propNode) == 100 );

        *propNode = 101;

        REQUIRE( (int)(*propNode) == 101 );

        // Script Test
        staticSerializableObject.Script().RunString("CF:GetProperty('testNameStatic/testNameContainerStatic/node[0].Property1').Number = 1");
        staticSerializableObject.Script().RunString("CF:GetProperty('testNameStatic/testNameContainerStatic/node[0].Property2').Number = 789");

        REQUIRE( (int)(*propNode) == 1 );

        propNode = staticSerializableObject.PropertyManager().GetPropertyFromPath( "testNameStatic/testNameContainerStatic/node[0].Property2" );

        REQUIRE( (int)(*propNode) == 789 );

        // Property access by selection
        propNode = staticSerializableObject.PropertyManager().GetPropertyFromPath( "testNameStatic/testNameContainerStatic/node[*].Property1" );

        REQUIRE( smart_ptr_isValid( propNode ) );

        INFO ( "The selection property name: " << propNode->Name() );

        //*propNode = 777;
    }
}
