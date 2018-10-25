#ifndef CONSTELEMENT_HPP_INCLUDED
#define CONSTELEMENT_HPP_INCLUDED

#include <serializable.hpp>

#include "physicsbody.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
class ConstElement : public PhysicsBody
{
        CODEFRAME_META_CLASS_NAME( "ConstElement" );
        CODEFRAME_META_BUILD_ROLE( codeframe::OBJECT );
        CODEFRAME_META_BUILD_TYPE( codeframe::DYNAMIC );

    public:
        ConstElement( std::string name );
        virtual ~ConstElement();
        ConstElement(const ConstElement& other);
        ConstElement& operator=(const ConstElement& rhs);
};

#endif // CONSTELEMENT_HPP_INCLUDED
