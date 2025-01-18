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

        ~classTest_Static() override
        {

        }
};

class classTest_SubInternal : public codeframe::Object
{
    public:
        CODEFRAME_META_CLASS_NAME( "classTest_SubInternal" );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

        codeframe::Property<int> Property1;
        codeframe::Property<int> Property2;
        codeframe::Property<int> Property3;
        codeframe::Property<int> Property4;
        codeframe::Property<int> Property_rew;
        codeframe::Property<float> Property_float;

        classTest_SubInternal( const std::string& name, ObjectNode* parent ) : Object( name, parent ),
            Property1( this, "Property1", 8100U , cPropertyInfo().Kind( KIND_NUMBER ).Description("Property1_desc") ),
            Property2( this, "Property2", 8200U , cPropertyInfo().Kind( KIND_NUMBER ).Description("Property2_desc") ),
            Property3( this, "Property3", 8300U , cPropertyInfo().Kind( KIND_NUMBER ).Description("Property3_desc") ),
            Property4( this, "Property4", 8400U , cPropertyInfo().Kind( KIND_NUMBER ).Description("Property4_desc") ),
            Property_rew( this, "Property_rew", 8400U , cPropertyInfo().Kind( KIND_NUMBER ).Description("Property_rew_desc") ),
            Property_float( this, "Property_float", 81.2f , cPropertyInfo().Kind( KIND_REAL ).Description("Property_float") )
        {
        }

        ~classTest_SubInternal() override
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
        codeframe::Property<int> Property_rew;
        codeframe::Property<float> Property_float;

        codeframe::Property<int> PropertyLink;
        codeframe::Property<int> PropertyLink_rel;

        classTest_SubInternal InternalObject;

    public:
        classTest_Dynamic( const std::string& name, ObjectNode* parent ) :
            Object( name, parent ),
            Property1( this, "Property1", 100U , cPropertyInfo().Kind( KIND_NUMBER ).Description("Property1_desc") ),
            Property2( this, "Property2", 200U , cPropertyInfo().Kind( KIND_NUMBER ).Description("Property2_desc") ),
            Property3( this, "Property3", 300U , cPropertyInfo().Kind( KIND_NUMBER ).Description("Property3_desc") ),
            Property4( this, "Property4", 400U , cPropertyInfo().Kind( KIND_NUMBER ).Description("Property4_desc") ),
            Property_rew( this, "Property_rew", 400U , cPropertyInfo().Kind( KIND_NUMBER ).Description("Property_rew_desc") ),
            Property_float( this, "Property_float", 1.2f , cPropertyInfo().Kind( KIND_REAL ).Description("Property_float") ),

            PropertyLink( this, "PropertyLink", 500U,
                            cPropertyInfo().
                                Kind( KIND_NUMBER ).
                                ReferencePath("testNameStatic/testNameContainerStatic/node[0].Property1").
                                Description("Property4_desc") ),

            PropertyLink_rel( this, "PropertyLink_rel", 600U,
                cPropertyInfo().
                    Kind( KIND_NUMBER ).
                    ReferencePath("/../node[0].Property1").
                    Description("PropertyLink_rel_desc") ),
            InternalObject("InternalObject", this)
        {
        }

        ~classTest_Dynamic() override
        {

        }
};

class classTest_Dynamic_rel : public codeframe::Object
{
    public:
        CODEFRAME_META_CLASS_NAME( "classTest_Dynamic_rel" );
        CODEFRAME_META_BUILD_TYPE( codeframe::DYNAMIC );

        codeframe::Property<int> PropertyLink_rel;
        codeframe::Property<int> PropertyLink_rel_rew;

    public:
        classTest_Dynamic_rel( const std::string& name, ObjectNode* parent ) :
            Object( name, parent ),
            PropertyLink_rel( this, "PropertyLink_rel", 600U,
                cPropertyInfo().
                    Kind( KIND_NUMBER ).
                    ReferencePath("/../node[4].Property1").
                    Description("PropertyLink_rel_desc") ),
            PropertyLink_rel_rew( this, "PropertyLink_rel_rew", 700U,
                cPropertyInfo().
                    Kind( KIND_NUMBER ).
                    ReferencePath("/../node[4].Property_rew>").
                    Description("PropertyLink_rel_rew") )
        {
        }
};

class classTest_Container : public codeframe::ObjectContainer
{
    public:
        CODEFRAME_META_CLASS_NAME( "classTest_Container" );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

        classTest_Container( const std::string& name, smart_ptr<ObjectNode> parent ) :
            ObjectContainer( name, parent )
        {
        }

        virtual smart_ptr<codeframe::Object> Create(
                                                    const std::string& className,
                                                    const std::string& objName,
                                                    const std::vector<codeframe::VariantValue>& params = std::vector<codeframe::VariantValue>()
                                                 )
        {
            if (className == "classTest_Dynamic")
            {
                smart_ptr<classTest_Dynamic> obj = smart_ptr<classTest_Dynamic>(new classTest_Dynamic(objName, this));

                InsertObject(obj);

                return smart_ptr<codeframe::Object>(obj);
            }
            else if (className == "classTest_Dynamic_rel")
            {
                smart_ptr<classTest_Dynamic_rel> obj = smart_ptr<classTest_Dynamic_rel>(new classTest_Dynamic_rel(objName, this));

                InsertObject(obj);

                return smart_ptr<codeframe::Object>(obj);
            }

            return smart_ptr<codeframe::Object>();
        }

    public:
        //using codeframe::ObjectContainer::ObjectContainer;
};

class classTestStaticDynamic_Container : public codeframe::ObjectContainer
{
    public:
        CODEFRAME_META_CLASS_NAME( "classTestStaticDynamic_Container" );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

        classTestStaticDynamic_Container( const std::string& name, smart_ptr<ObjectNode> parent ) :
            ObjectContainer( name, parent ),
            StaticInternalObject("StaticInternalObject", this)
        {
        }

        ~classTestStaticDynamic_Container()
        {
        }

        classTest_SubInternal StaticInternalObject;

        virtual smart_ptr<codeframe::Object> Create(
                                                    const std::string& className,
                                                    const std::string& objName,
                                                    const std::vector<codeframe::VariantValue>& params = std::vector<codeframe::VariantValue>()
                                                 )
        {
            if (className == "classTest_Dynamic")
            {
                smart_ptr<classTest_Dynamic> obj = smart_ptr<classTest_Dynamic>(new classTest_Dynamic(objName, this));

                InsertObject(obj);

                return smart_ptr<codeframe::Object>(obj);
            }
            else if (className == "classTest_Dynamic_rel")
            {
                smart_ptr<classTest_Dynamic_rel> obj = smart_ptr<classTest_Dynamic_rel>(new classTest_Dynamic_rel(objName, this));

                InsertObject(obj);

                return smart_ptr<codeframe::Object>(obj);
            }

            return smart_ptr<codeframe::Object>();
        }

    public:
        //using codeframe::ObjectContainer::ObjectContainer;
};

#endif // TEST_CODEFRAME_FIXTURE_HPP_INCLUDED
