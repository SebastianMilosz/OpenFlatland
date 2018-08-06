#ifndef CONSTELEMENT_HPP_INCLUDED
#define CONSTELEMENT_HPP_INCLUDED

#include <serializable.h>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
class ConstElement : public codeframe::cSerializable
{
    public:
        std::string Role()      const { return "Object";       }
        std::string Class()     const { return "ConstElement"; }
        std::string BuildType() const { return "Dynamic";      }

    public:
        ConstElement( std::string name );
        virtual ~ConstElement();
        ConstElement(const ConstElement& other);
        ConstElement& operator=(const ConstElement& other);
};

#endif // CONSTELEMENT_HPP_INCLUDED
