#include "serializable_property_selection.hpp"

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase& PropertySelection::operator=( const PropertySelection& val )
    {
        m_Mutex.Lock();
        val.m_Mutex.Lock();

        m_Mutex.Unlock();
        val.m_Mutex.Unlock();

        return *this;
    }

    bool PropertySelection::operator==(const PropertySelection& sval) const
    {
        return false;
    }

    bool PropertySelection::operator!=(const PropertySelection& sval) const
    {
        return false;
    }

    bool PropertySelection::operator==(const int& sval) const
    {
        return false;
    }

    bool PropertySelection::operator!=(const int& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase& PropertySelection::operator=(bool val)
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase& PropertySelection::operator=(char val)
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase& PropertySelection::operator=(unsigned char val)
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase& PropertySelection::operator=(int val)
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase& PropertySelection::operator=(unsigned int val)
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase& PropertySelection::operator=(float val)
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase& PropertySelection::operator=(double val)
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase& PropertySelection::operator=( const std::string& val )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase& PropertySelection::operator++()
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase& PropertySelection::operator--()
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase& PropertySelection::operator+=( const PropertySelection& rhs )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase& PropertySelection::operator-=( const PropertySelection& rhs )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase PropertySelection::operator+(const PropertySelection& rhs)
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase PropertySelection::operator-( const PropertySelection& rhs )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase& PropertySelection::operator+=( const int rhs )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase& PropertySelection::operator-=( const int rhs )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertySelection::Name() const
    {
        return m_name;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertySelection::NameIs( const std::string& name ) const
    {
        if( name == m_name ) return true;
        return false;
    }
}
