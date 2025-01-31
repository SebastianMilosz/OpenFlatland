#ifndef SERIALIZABLE_PROPERTY_ITERATOR_HPP_INCLUDED
#define SERIALIZABLE_PROPERTY_ITERATOR_HPP_INCLUDED

#include <map>
#include <string>

namespace codeframe
{
    class PropertyBase;
    class cPropertyList;

    /*****************************************************************************/
    /**
      * @brief Bidirectional iterator for property list
     **
    ******************************************************************************/
    class PropertyIterator
    {
        using iterator_category = std::input_iterator_tag;
        using value_type = PropertyBase*;
        using difference_type = std::ptrdiff_t;
        using pointer = PropertyBase**;
        using reference = PropertyBase*&;

        friend class cPropertyList;

    public:
        PropertyIterator(const PropertyIterator& n);

        PropertyBase*     operator *();
        PropertyIterator& operator ++();

        bool operator==(const PropertyIterator& n);
        bool operator!=(const PropertyIterator& n);

    private:
        PropertyIterator( std::map<std::string, PropertyBase*>::iterator iter );

        std::map<std::string, PropertyBase*>::iterator m_iterator;
    };
}

#endif // SERIALIZABLE_PROPERTY_ITERATOR_HPP_INCLUDED
