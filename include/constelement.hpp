#ifndef CONSTELEMENT_HPP_INCLUDED
#define CONSTELEMENT_HPP_INCLUDED

#include <serializable.h>

#include "physicsbody.h"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
class ConstElement : public PhysicsBody
{
    public:
        std::string Role()            const { return "Object";       }
        std::string Class()           const { return "ConstElement"; }
        std::string BuildType()       const { return "Dynamic";      }
        std::string ConstructPatern() const { return ""; }

    public:
        ConstElement( std::string name );
        virtual ~ConstElement();
        ConstElement(const ConstElement& other);
        ConstElement& operator=(const ConstElement& rhs);
};

#endif // CONSTELEMENT_HPP_INCLUDED
