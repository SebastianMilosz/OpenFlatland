#include "serializable_property_selection.hpp"

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection::PropertySelection( PropertyNode* prop )
    {
        if ( prop != nullptr )
        {
            m_selection = prop;
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection::~PropertySelection()
    {

    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator=( const PropertySelection& val )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertySelection::operator==(const PropertySelection& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertySelection::operator!=(const PropertySelection& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertySelection::operator==(const int& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertySelection::operator!=(const int& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator=(bool val)
    {
        *m_selection = val;
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator=(char val)
    {
        *m_selection = val;
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator=(unsigned char val)
    {
        *m_selection = val;
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator=(int val)
    {
        *m_selection = val;
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator=(unsigned int val)
    {
        *m_selection = val;
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator=(float val)
    {
        *m_selection = val;
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator=(double val)
    {
        *m_selection = val;
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator=( const std::string& val )
    {
        *m_selection = val;
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator++()
    {
        ++(*m_selection);
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator--()
    {
        --(*m_selection);
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator+=( const PropertySelection& rhs )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator-=( const PropertySelection& rhs )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator+(const PropertySelection& rhs)
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator-( const PropertySelection& rhs )
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
        return m_selection->Name();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertySelection::NameIs( const std::string& name ) const
    {
        std::string thisName = Name();
        if ( thisName.compare( name ) == 0 )
        {
            return true;
        }
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertySelection::ToString()
    {
        return m_selection->ToString();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    eType PropertySelection::Type() const
    {
        return m_selection->Type();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode* PropertySelection::Reference() const
    {
        return m_selection->Reference();
    }

    uint32_t PropertySelection::Id() const
    {
        return m_selection->Id();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    ObjectNode* PropertySelection::Parent() const
    {
        return m_selection->Parent();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertySelection::ConnectReference( smart_ptr<PropertyNode> refNode )
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection::operator bool() const
    {
        return (bool)(*m_selection);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection::operator char() const
    {
        return (char)(*m_selection);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection::operator unsigned char() const
    {
        return (unsigned char)(*m_selection);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection::operator int() const
    {
        return (int)(*m_selection);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection::operator unsigned int() const
    {
        return (unsigned int)(*m_selection);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection::operator unsigned short() const
    {
        return (unsigned short)(*m_selection);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection::operator double() const
    {
        return (double)(*m_selection);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection::operator float() const
    {
        return (float)(*m_selection);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection::operator std::string() const
    {
        return (std::string)(*m_selection);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertySelection::SetNumber( const int val )
    {
        m_selection->SetNumber( val );
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int PropertySelection::GetNumber() const
    {
        return m_selection->GetNumber();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertySelection::SetReal( const double val )
    {
        m_selection->SetReal( val );
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    double PropertySelection::GetReal() const
    {
        return m_selection->GetReal();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertySelection::SetString( const std::string&  val )
    {
        m_selection->SetString( val );
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertySelection::GetString() const
    {
        return m_selection->GetString();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertySelection::Lock() const
    {
        m_selection->Lock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertySelection::Unlock() const
    {
        m_selection->Unlock();
    }
}
