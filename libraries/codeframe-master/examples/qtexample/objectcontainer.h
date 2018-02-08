#ifndef OBJECTCONTAINER_H
#define OBJECTCONTAINER_H

#include <serializable.h>
#include <serializablecontainer.h>

class ObjectContainer : public codeframe::cSerializableContainer<codeframe::cSerializable>
{
public:
    std::string Role()      { return "Container"; }
    std::string Class()     { return "cSerializableContainer"; }
    std::string BuildType() { return "Static"; }

public:
    ObjectContainer();

    smart::ptr<codeframe::cSerializable> Create( std::string className, std::string objName, int cnt = -1 );
};

#endif // OBJECTCONTAINER_H
