#include "serializable_property_iterator.hpp"

#include "serializable_property_list.hpp"

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyIterator::PropertyIterator(const PropertyIterator& n) :
        m_iterator(n.m_iterator)
    {
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase* PropertyIterator::operator *()
    {
        return m_iterator->second;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyIterator& PropertyIterator::operator ++()
    {
        ++m_iterator;
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertyIterator::operator==(const PropertyIterator& n)
    {
        return n.m_iterator == m_iterator;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertyIterator::operator!=(const PropertyIterator& n)
    {
        return !(n.m_iterator == m_iterator);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyIterator::PropertyIterator(std::map<std::string, PropertyBase*>::iterator iter) :
        m_iterator( iter )
    {
    }
}
