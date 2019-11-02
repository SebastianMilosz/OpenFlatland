#include "catch.hpp"

#include <serializable.hpp>
#include <serializablecontainer.hpp>

using namespace codeframe;

TEST_CASE( "Serializable library path", "[serializable.Path]" )
{
    class classTest_Static : public codeframe::cSerializable
    {
        public:
            CODEFRAME_META_CLASS_NAME( "classTest_Static" );
            CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

        public:
            classTest_Static( const std::string& name, cSerializableInterface* parent ) : cSerializable( name, parent )
            {
            }
    };

    class classTest_Dynamic : public codeframe::cSerializable
    {
        public:
            CODEFRAME_META_CLASS_NAME( "classTest_Dynamic" );
            CODEFRAME_META_BUILD_TYPE( codeframe::DYNAMIC );

        codeframe::Property<int> Property1;
        codeframe::Property<int> Property2;
        codeframe::Property<int> Property3;
        codeframe::Property<int> Property4;

        public:
            classTest_Dynamic( const std::string& name, cSerializableInterface* parent ) :
                cSerializable( name, parent ),
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

            virtual smart_ptr<cSerializableInterface> Create(
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

                return smart_ptr<codeframe::cSerializableInterface>();
            }

        public:
            classTest_Container( const std::string& name, cSerializableInterface* parent ) : cSerializableContainer( name, parent )
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

    SECTION( "Basic serializable objects tests" )
    {
        REQUIRE( staticSerializableObject.Identity().ObjectName() == "testNameStatic" );
        REQUIRE( staticContainerObject.Identity().ObjectName() == "testNameContainerStatic" );

        REQUIRE( staticContainerObject.Count() == 5 );

        REQUIRE( (int)(*static_cast<classTest_Dynamic*>(smart_ptr_getRaw(staticContainerObject[0]))).Property1 == 100 );

        // Direct property access
        PropertyBase* prop = staticSerializableObject.PropertyManager().GetPropertyFromPath( "testNameStatic/testNameContainerStatic/node[0].Property1" );
        REQUIRE( prop != NULL );
        REQUIRE( (int)(*prop) == 100 );

        *prop = 101;

        REQUIRE( (int)(*prop) == 101 );

        // Property access by selection
        prop = staticSerializableObject.PropertyManager().GetPropertyFromPath( "testNameStatic/testNameContainerStatic/node[*].Property1" );

        REQUIRE( prop != NULL );

        //*prop = 777;
    }
}
