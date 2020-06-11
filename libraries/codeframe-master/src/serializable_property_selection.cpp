#include "serializable_property_selection.hpp"

#include <cassert>

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection::PropertySelection( PropertyNode* prop ) :
        m_selection( prop )
    {
        assert( prop );
        prop->signalDeleted.connect(this, &PropertySelection::OnDelete);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection::PropertySelection( const PropertySelection& prop ) :
        m_selection(prop.m_selection)
    {
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
    void PropertySelection::OnDelete(void* deletedPtr)
    {
        m_selection = nullptr;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection& PropertySelection::operator=( const PropertySelection& val )
    {
        m_selection = val.m_selection;
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertySelection::operator==(const PropertyNode& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertySelection::operator!=(const PropertyNode& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertySelection::operator==(const int& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertySelection::operator!=(const int& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator=(const bool_t val)
    {
        if (m_selection)
        {
            *m_selection = val;
        }
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator=(const char val)
    {
        if (m_selection)
        {
            *m_selection = val;
        }
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator=(const unsigned char val)
    {
        if (m_selection)
        {
            *m_selection = val;
        }
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator=(const int val)
    {
        if (m_selection)
        {
            *m_selection = val;
        }
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator=(const unsigned int val)
    {
        if (m_selection)
        {
            *m_selection = val;
        }
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator=(const float val)
    {
        if (m_selection)
        {
            *m_selection = val;
        }
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator=(const double val)
    {
        if (m_selection)
        {
            *m_selection = val;
        }
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator=(const std::string& val)
    {
        if (m_selection)
        {
            *m_selection = val;
        }
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator++()
    {
        if (m_selection)
        {
            ++(*m_selection);
        }
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator--()
    {
        if (m_selection)
        {
            --(*m_selection);
        }
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator+=(const int rhs)
    {
        if (m_selection)
        {
            (*m_selection) += rhs;
        }
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator-=(const int rhs)
    {
        if (m_selection)
        {
            (*m_selection) -= rhs;
        }
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator+=( const PropertyNode& rhs )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator-=( const PropertyNode& rhs )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator+(const PropertyNode& rhs)
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertySelection::operator-( const PropertyNode& rhs )
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
        if (m_selection)
        {
            return m_selection->Name();
        }
        return "PropertySelection\nullptr";
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertySelection::NameIs( const std::string& name ) const
    {
        std::string thisName( Name() );
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
    std::string PropertySelection::ToString() const
    {
        if (m_selection)
        {
            return m_selection->ToString();
        }
        return "PropertySelection\nullptr";
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    eType PropertySelection::Type() const
    {
        if (m_selection)
        {
            return m_selection->Type();
        }
        return eType::TYPE_NON;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertySelection::Path( bool_t addName ) const
    {
        if (m_selection)
        {
            return m_selection->Path( addName );
        }
        return "PropertySelection\nullptr";
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    smart_ptr<PropertyNode> PropertySelection::Reference() const
    {
        if (m_selection)
        {
            return m_selection->Reference();
        }
        return smart_ptr<PropertyNode>(nullptr);
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    uint32_t PropertySelection::Id() const
    {
        if (m_selection)
        {
            return m_selection->Id();
        }
        return 0U;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    smart_ptr<ObjectNode> PropertySelection::Parent() const
    {
        if (m_selection)
        {
            return m_selection->Parent();
        }
        return nullptr;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertySelection::ParentName() const
    {
        if (m_selection)
        {
            return m_selection->ParentName();
        }
        return "PropertySelection\nullptr";
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertySelection::ConnectReference( smart_ptr<PropertyNode> refNode )
    {
        if (m_selection)
        {
            return m_selection->ConnectReference(refNode);
        }
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertySelection::TypeString() const
    {
        if (m_selection)
        {
            return m_selection->TypeString();
        }
        return "PropertySelection\nullptr";
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertySelection::PreviousValueString() const
    {
        if (m_selection)
        {
            return m_selection->PreviousValueString();
        }
        return "PropertySelection\nullptr";
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertySelection::CurentValueString() const
    {
        if (m_selection)
        {
            return m_selection->CurentValueString();
        }
        return "PropertySelection\nullptr";
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int PropertySelection::PreviousValueInteger() const
    {
        if (m_selection)
        {
            return m_selection->PreviousValueInteger();
        }
        return 0;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int PropertySelection::CurentValueInteger() const
    {
        if (m_selection)
        {
            return m_selection->CurentValueInteger();
        }
        return 0;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection::operator bool() const
    {
        if (m_selection)
        {
            return (bool)(*m_selection);
        }
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection::operator char() const
    {
        if (m_selection)
        {
            return (char)(*m_selection);
        }
        return 0;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection::operator unsigned char() const
    {
        if (m_selection)
        {
            return (unsigned char)(*m_selection);
        }
        return 0U;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection::operator int() const
    {
        if (m_selection)
        {
            return (int)(*m_selection);
        }
        return 0;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection::operator unsigned int() const
    {
        if (m_selection)
        {
            return (unsigned int)(*m_selection);
        }
        return 0U;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection::operator unsigned short() const
    {
        if (m_selection)
        {
            return (unsigned short)(*m_selection);
        }
        return 0U;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection::operator double() const
    {
        if (m_selection)
        {
            return (double)(*m_selection);
        }
        return 0;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection::operator float() const
    {
        if (m_selection)
        {
            return (float)(*m_selection);
        }
        return 0.0F;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertySelection::operator std::string() const
    {
        if (m_selection)
        {
            return (std::string)(*m_selection);
        }
        return "PropertySelection\nullptr";
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertySelection::IsReference() const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int PropertySelection::ToInt() const
    {
        if (m_selection)
        {
            return m_selection->GetNumber();
        }
        return 0;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertySelection::SetNumber( const int val )
    {
        if (m_selection)
        {
            m_selection->SetNumber( val );
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int PropertySelection::GetNumber() const
    {
        if (m_selection)
        {
            return m_selection->GetNumber();
        }
        return 0;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertySelection::SetReal( const double val )
    {
        if (m_selection)
        {
            m_selection->SetReal( val );
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    double PropertySelection::GetReal() const
    {
        if (m_selection)
        {
            return m_selection->GetReal();
        }
        return 0;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertySelection::SetString( const std::string&  val )
    {
        if (m_selection)
        {
            m_selection->SetString( val );
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertySelection::GetString() const
    {
        if (m_selection)
        {
            return m_selection->GetString();
        }
        return "PropertySelection\nullptr";
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertySelection::Lock() const
    {
        if (m_selection)
        {
            m_selection->Lock();
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertySelection::Unlock() const
    {
        if (m_selection)
        {
            m_selection->Unlock();
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertySelection::IsChanged() const
    {
        if (m_selection)
        {
            return m_selection->IsChanged();
        }
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertySelection::EmitChanges()
    {
        if (m_selection)
        {
            m_selection->EmitChanges();
        }
    }
}
