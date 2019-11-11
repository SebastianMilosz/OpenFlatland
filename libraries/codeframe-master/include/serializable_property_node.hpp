#ifndef _SERIALIZABLE_PROPERTY_NODE_H
#define _SERIALIZABLE_PROPERTY_NODE_H

#include "typeinfo.hpp"

#include <string>
#include <smartpointer.h>

namespace codeframe
{
    class cSerializableInterface;

    /*****************************************************************************
     * @class Interface for all properties classes
     *****************************************************************************/
    class PropertyNode
    {
        public:
            virtual PropertyNode& operator=(const bool          val) = 0;
            virtual PropertyNode& operator=(const char          val) = 0;
            virtual PropertyNode& operator=(const unsigned char val) = 0;
            virtual PropertyNode& operator=(const int           val) = 0;
            virtual PropertyNode& operator=(const unsigned int  val) = 0;
            virtual PropertyNode& operator=(const float         val) = 0;
            virtual PropertyNode& operator=(const double        val) = 0;
            virtual PropertyNode& operator=(const std::string&  val) = 0;
            virtual PropertyNode& operator++() = 0;
            virtual PropertyNode& operator--() = 0;

            virtual operator bool() const = 0;
            virtual operator char() const = 0;
            virtual operator unsigned char() const = 0;
            virtual operator int() const = 0;
            virtual operator unsigned int() const = 0;
            virtual operator unsigned short() const = 0;
            virtual operator double() const = 0;
            virtual operator float() const = 0;
            virtual operator std::string() const = 0;

            virtual std::string ToString() = 0;
            virtual std::string Name() const = 0;
            virtual bool        NameIs( const std::string& name ) const = 0;
            virtual eType       Type() const = 0;
            virtual PropertyNode* Reference() const = 0;
            virtual uint32_t      Id() const = 0;

            virtual cSerializableInterface* Parent() const = 0;

            virtual bool        ConnectReference( smart_ptr<PropertyNode> refNode ) = 0;

            virtual void        SetNumber( const int val ) = 0;
            virtual int         GetNumber() const = 0;
            virtual void        SetReal( const double val ) = 0;
            virtual double      GetReal() const = 0;
            virtual void        SetString( const std::string&  val ) = 0;
            virtual std::string GetString() const = 0;

            virtual Lock() const = 0;
            virtual Unlock() const = 0;
    };
}

#endif //_SERIALIZABLE_PROPERTY_NODE_H