#ifndef SERIALIZABLE_PROPERTY_SELECTION_HPP_INCLUDED
#define SERIALIZABLE_PROPERTY_SELECTION_HPP_INCLUDED

#include "serializablepropertybase.hpp"

#include <string>
#include <vector>

namespace codeframe
{
     /*****************************************************************************
     * @class This class stores Property's from selection
     *****************************************************************************/
    class PropertySelection : public PropertyBase
    {
            virtual std::string     Name() const;
            virtual bool            NameIs( const std::string& name ) const;

            //
            virtual bool operator==(const PropertySelection& sval) const;
            virtual bool operator!=(const PropertySelection& sval) const;

            virtual bool operator==(const int& sval) const;
            virtual bool operator!=(const int& sval) const;

            //
            virtual PropertyBase& operator=(const PropertySelection& val);
            virtual PropertyBase& operator=(const bool          val);
            virtual PropertyBase& operator=(const char          val);
            virtual PropertyBase& operator=(const unsigned char val);
            virtual PropertyBase& operator=(const int           val);
            virtual PropertyBase& operator=(const unsigned int  val);
            virtual PropertyBase& operator=(const float         val);
            virtual PropertyBase& operator=(const double        val);
            virtual PropertyBase& operator=(const std::string&  val);
            virtual PropertyBase& operator++();
            virtual PropertyBase& operator--();
            virtual PropertyBase& operator+=(const PropertySelection& rhs);
            virtual PropertyBase& operator-=(const PropertySelection& rhs);
            virtual PropertyBase  operator+ (const PropertySelection& rhs);
            virtual PropertyBase  operator- (const PropertySelection& rhs);
            virtual PropertyBase& operator+=(const int rhs);
            virtual PropertyBase& operator-=(const int rhs);

        private:
            std::vector<PropertyBase*> m_selectionVector;
    };
}

#endif // SERIALIZABLE_PROPERTY_SELECTION_HPP_INCLUDED
