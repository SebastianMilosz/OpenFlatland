#ifndef SERIALIZABLE_PROPERTY_SELECTION_HPP_INCLUDED
#define SERIALIZABLE_PROPERTY_SELECTION_HPP_INCLUDED

#include "serializable_property_node.hpp"

#include <string>
#include <vector>

namespace codeframe
{
     /*****************************************************************************
     * @class This class stores Property's from selection
     *****************************************************************************/
    class PropertySelection : public PropertyNode
    {
        public:
            PropertySelection( PropertyNode* prop );
           ~PropertySelection();

            void Add( PropertyNode* prop );

            virtual std::string Name() const;
            virtual bool        NameIs( const std::string& name ) const;

            virtual bool ConnectReference( smart_ptr<PropertyNode> refNode );

            virtual bool operator==(const PropertySelection& sval) const;
            virtual bool operator!=(const PropertySelection& sval) const;

            virtual bool operator==(const int& sval) const;
            virtual bool operator!=(const int& sval) const;

            virtual PropertyNode& operator=(const PropertySelection& val);
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
            virtual PropertyNode& operator+=(const PropertySelection& rhs);
            virtual PropertyNode& operator-=(const PropertySelection& rhs);
            virtual PropertyNode& operator+ (const PropertySelection& rhs);
            virtual PropertyNode& operator- (const PropertySelection& rhs);

        private:
            std::vector<PropertyNode*> m_selectionVector;
    };
}

#endif // SERIALIZABLE_PROPERTY_SELECTION_HPP_INCLUDED
