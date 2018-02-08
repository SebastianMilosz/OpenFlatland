#include "objectcontainer.h"
#include "objectchild.h"

ObjectContainer::ObjectContainer() :
    cSerializableContainer( "ObjectContainer", NULL )
{
}

smart::ptr<codeframe::cSerializable> ObjectContainer::Create( std::string className, std::string objName, int cnt )
{
    if( className == "ObjectChild" )
    {
        smart::ptr<codeframe::cSerializable> obj = smart::ptr<codeframe::cSerializable>( new ObjectChild( objName, (codeframe::cSerializable*)this ) );

        InsertObject( obj, cnt );
        return obj;
    }

    return smart::ptr<codeframe::cSerializable>();
}
