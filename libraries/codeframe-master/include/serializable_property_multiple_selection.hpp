#ifndef SERIALIZABLE_PROPERTY_MULTIPLE_SELECTION_HPP_INCLUDED
#define SERIALIZABLE_PROPERTY_MULTIPLE_SELECTION_HPP_INCLUDED

#include "serializable_property_node.hpp"

#include <string>
#include <vector>

namespace codeframe
{
     /*****************************************************************************
     * @class This class stores Property's from selection
     *****************************************************************************/
    class PropertyMultipleSelection : public PropertyNode
    {
        public:
            PropertyMultipleSelection();
           ~PropertyMultipleSelection();

            void Add( smart_ptr<PropertyNode> prop );

            virtual std::string Name() const;
            virtual bool        NameIs( const std::string& name ) const;
            virtual std::string ToString();
            virtual eType       Type() const;
            virtual PropertyNode* Reference() const;
            virtual uint32_t      Id() const;

            virtual ObjectNode* Parent() const;
            virtual std::string ParentName() const;
            virtual bool ConnectReference( smart_ptr<PropertyNode> refNode );

            virtual bool operator==(const PropertyMultipleSelection& sval) const;
            virtual bool operator!=(const PropertyMultipleSelection& sval) const;

            virtual bool operator==(const int& sval) const;
            virtual bool operator!=(const int& sval) const;

            virtual PropertyNode& operator=(const PropertyMultipleSelection& val);
            virtual PropertyNode& operator=(const bool          val);
            virtual PropertyNode& operator=(const char          val);
            virtual PropertyNode& operator=(const unsigned char val);
            virtual PropertyNode& operator=(const int           val);
            virtual PropertyNode& operator=(const unsigned int  val);
            virtual PropertyNode& operator=(const float         val);
            virtual PropertyNode& operator=(const double        val);
            virtual PropertyNode& operator=(const std::string&  val);
            virtual PropertyNode& operator++();
            virtual PropertyNode& operator--();
            virtual PropertyNode& operator+=(const PropertyMultipleSelection& rhs);
            virtual PropertyNode& operator-=(const PropertyMultipleSelection& rhs);
            virtual PropertyNode& operator+ (const PropertyMultipleSelection& rhs);
            virtual PropertyNode& operator- (const PropertyMultipleSelection& rhs);

            virtual operator bool() const;
            virtual operator char() const;
            virtual operator unsigned char() const;
            virtual operator int() const;
            virtual operator unsigned int() const;
            virtual operator unsigned short() const;
            virtual operator double() const;
            virtual operator float() const;
            virtual operator std::string() const;

            virtual void        SetNumber( const int val );
            virtual int         GetNumber() const;
            virtual void        SetReal( const double val );
            virtual double      GetReal() const;
            virtual void        SetString( const std::string&  val );
            virtual std::string GetString() const;

            virtual void Lock() const;
            virtual void Unlock() const;
        private:
            std::vector< smart_ptr<PropertyNode> > m_selectionVector;
    };
}

#endif // SERIALIZABLE_PROPERTY_MULTIPLE_SELECTION_HPP_INCLUDED
