#ifndef OBJECTCHILD_H
#define OBJECTCHILD_H

#include <serializable.h>

class ObjectChild : public codeframe::cSerializable
{
public:
    std::string Role()      { return "Object"; }
    std::string Class()     { return "ObjectChild"; }
    std::string BuildType() { return "Dynamic"; }

public:
    ObjectChild(std::string name, cSerializable* parent);
};

#endif // OBJECTCHILD_H
