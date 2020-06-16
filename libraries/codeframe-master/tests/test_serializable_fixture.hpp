#ifndef TEST_CODEFRAME_FIXTURE_HPP_INCLUDED
#define TEST_CODEFRAME_FIXTURE_HPP_INCLUDED

#include <serializable_object.hpp>
#include <serializable_object_container.hpp>

using namespace codeframe;

class classTest_Static : public codeframe::Object
{
    public:
        CODEFRAME_META_CLASS_NAME( "classTest_Static" );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

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
        using codeframe::ObjectContainer::ObjectContainer;
};

#endif // TEST_CODEFRAME_FIXTURE_HPP_INCLUDED
