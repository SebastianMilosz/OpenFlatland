#ifndef CONSTELEMENT_HPP_INCLUDED
#define CONSTELEMENT_HPP_INCLUDED

#include <serializable_object.hpp>

#include "physics_body.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
class ConstElement : public PhysicsBody
{
        CODEFRAME_META_CLASS_NAME( "ConstElement" );
        CODEFRAME_META_BUILD_TYPE( codeframe::DYNAMIC );

    public:
        ConstElement( const std::string& name );
        virtual ~ConstElement() = default;

        ConstElement(const ConstElement& other);
        ConstElement& operator=(const ConstElement& rhs);
};

#endif // CONSTELEMENT_HPP_INCLUDED
