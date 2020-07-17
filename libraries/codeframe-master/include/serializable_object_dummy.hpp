#ifndef SERIALIZABLE_OBJECT_DUMMY_HPP_INCLUDED
#define SERIALIZABLE_OBJECT_DUMMY_HPP_INCLUDED

#include "serializable_object.hpp"

namespace codeframe
{
    class ObjectDummy : public Object
    {
            CODEFRAME_META_CLASS_NAME( "ObjectDummy" );
            CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

        public:
            ObjectDummy( const std::string& name ) : Object(name) {}
            virtual ~ObjectDummy() = default;
    };
}

#endif // SERIALIZABLE_OBJECT_DUMMY_HPP_INCLUDED
