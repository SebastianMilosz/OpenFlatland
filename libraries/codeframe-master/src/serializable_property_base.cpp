#include "serializable_property_base.hpp"

#include <exception>
#include <stdexcept>
#include <sstream>
#include <algorithm>

#include "instance_manager.hpp"
#include "serializable_object.hpp"

namespace codeframe
{
    int PropertyBase::s_globalParConCnt = 0;

    /*****************************************************************************/
    /**
      * @brief Zamienia napis na liczbe 32b
     **
    ******************************************************************************/
    uint32_t PropertyBase::GetHashId( const std::string& str, uint16_t mod )
    {
        uint32_t hash, i;
        for ( hash = i = 0; i < str.size(); ++i )
        {
            hash += str[i];
            hash += (hash << 10);
            hash ^= (hash >> 6);
        }
        hash += (hash << 3);
        hash ^= (hash >> 11);
        hash += str.size();
        hash &= 0xFFFF0000;
        hash |= mod;

        return hash;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase::PropertyBase( ObjectNode* parentpc, const std::string& name, eType type, cPropertyInfo info ) :
        m_reference(nullptr),
        m_referenceParent(nullptr),
        m_type(type),
        m_parentpc( parentpc ),
        m_name(name),
        m_id(0),
        m_isWaitForUpdate( false ),
        m_waitForUpdateCnt(0),
        m_propertyInfo( info, this ),
        m_temporary( false )
        {
            RegisterProperty();
        }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase::PropertyBase( const PropertyBase& sval ) :
        m_reference      (sval.m_reference),
        m_referenceParent(sval.m_referenceParent),
        m_type           (sval.m_type),
        m_parentpc       (sval.m_parentpc),
        m_name           (sval.m_name),
        m_id             (sval.m_id),
        m_propertyInfo   (sval.m_propertyInfo),
        m_temporary      ( true )
    {
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase::~PropertyBase()
    {
        if( m_temporary == false )
        {
            UnRegisterProperty();
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertyBase::RegisterProperty()
    {
        if ( m_parentpc == nullptr )
        {
            return;
        }

        m_Mutex.Lock();

        int size = m_parentpc->PropertyList().GetObjectFieldCnt();

        m_id = GetHashId( Name(), 255 * s_globalParConCnt + size );

        m_parentpc->PropertyList().RegisterProperty( this );

        ObjectNode* rootNode = m_parentpc->Path().GetRootObject()->GetNode();

        if (rootNode)
        {
            ReferenceManager::ResolveReferences( *rootNode );
        }

        s_globalParConCnt++;

        m_Mutex.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertyBase::UnRegisterProperty()
    {
        if( m_parentpc == nullptr )
        {
            return;
        }

        m_Mutex.Lock();

        m_parentpc->PropertyList().UnRegisterProperty( this );

        s_globalParConCnt--;

        m_Mutex.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyBase::operator=(const PropertyNode& val)
    {
        m_Mutex.Lock();
        val.Lock();

        m_reference = val.Reference();
        m_name      = val.Name();
        m_id        = val.Id();
        m_type      = val.Type();
        m_parentpc  = val.Parent();

        m_Mutex.Unlock();
        val.Unlock();

        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertyBase::operator==(const PropertyBase& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertyBase::operator!=(const PropertyBase& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertyBase::operator==(const int& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertyBase::operator!=(const int& sval) const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyBase::operator=(const bool_t val)
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyBase::operator=(const char val)
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyBase::operator=(const unsigned char val)
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyBase::operator=(const int val)
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyBase::operator=(const unsigned int val)
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyBase::operator=(const float val)
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyBase::operator=(const double val)
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyBase::operator=( const std::string& val )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyBase::operator++()
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyBase::operator--()
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyBase::operator+=( const PropertyNode& rhs )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyBase::operator-=( const PropertyNode& rhs )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyBase::operator+(const PropertyNode& rhs)
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyBase::operator-( const PropertyNode& rhs )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyBase::operator+=( const int rhs )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyBase::operator-=( const int rhs )
    {
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertyBase::Name() const
    {
        return m_name;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertyBase::NameIs(const std::string& name) const
    {
        if ( name == m_name )
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
    uint32_t PropertyBase::Id() const
    {
        return m_id;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    eType PropertyBase::Type() const
    {
        return m_type;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertyBase::PreviousValueString() const
    {
        return "unknown from base";
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertyBase::CurentValueString() const
    {
        return "unknown from base";
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int PropertyBase::PreviousValueInteger() const
    {
        return 0;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int PropertyBase::CurentValueInteger() const
    {
        return 0;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertyBase::TypeString() const
    {
        return "default type from base";
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertyBase::Path( bool_t addName ) const
    {
        std::string propPath;

        if ( smart_ptr_isValid(m_parentpc) )
        {
            propPath = m_parentpc->Path().PathString();
            if ( addName )
            {
                propPath += "." + Name();
            }
        }
        else
        {
            throw std::runtime_error( "PropertyBase::Path() NULL Parent Container" );
        }

        return propPath;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    smart_ptr<ObjectNode> PropertyBase::Parent() const
    {
        return m_parentpc;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertyBase::ParentName() const
    {
        if ( smart_ptr_isValid(m_parentpc) )
        {
            return m_parentpc->Identity().ObjectName();
        }
        return "nullptr";
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertyBase::WaitForUpdate( int time )
    {
        m_isWaitForUpdate  = true;
        m_waitForUpdateCnt = time;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertyBase::WaitForUpdatePulse()
    {
        if ( (m_isWaitForUpdate) && (m_waitForUpdateCnt > 0) )
        {
           m_waitForUpdateCnt--;

           if ( m_waitForUpdateCnt == 0 )
           {
              m_isWaitForUpdate = false;
              //m_parentpc->signalUpdateFail.Emit( m_id );
           }
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertyBase::ConnectReference( smart_ptr<PropertyNode> refNode )
    {
        // Sprawdzamy czy zgadza sie typ
        if ( smart_ptr_isValid( refNode ) && (this->Type() == refNode->Type()) )
        {
            m_Mutex.Lock();
            m_referenceParent = refNode->Parent();
            m_reference       = refNode;
            m_Mutex.Unlock();
            return true;
        }

        return false;
    }

    /*****************************************************************************/
    /**
      * @brief Commit changes
     **
    ******************************************************************************/
    void PropertyBase::CommitChanges()
    {
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertyBase::IsChanged() const
    {
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertyBase::PulseChanged()
    {
        EmitChanges();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyNode& PropertyBase::WatchdogGetValue( int time )
    {
        WaitForUpdate( time );
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertyBase::SetNumber( const int val )
    {
        if ( Info().GetEnable() == false )
        {
            return;
        }

        *this = (int)val;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int PropertyBase::GetNumber() const
    {
        PropertyBase prop = *this;
        return (int)prop;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertyBase::SetReal( const double val )
    {
        if ( Info().GetEnable() == false )
        {
            return;
        }

        *this = (double)val;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    double PropertyBase::GetReal() const
    {
        PropertyBase prop = *this;
        return (double)prop;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertyBase::SetString( const std::string& val )
    {
        if ( Info().GetEnable() == false )
        {
            return;
        }

        *this = (std::string)val;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertyBase::GetString() const
    {
        PropertyBase prop = *this;
        return (std::string)prop;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertyBase::Lock() const
    {
        m_Mutex.Lock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertyBase::Unlock() const
    {
        m_Mutex.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertyBase::ToString() const
    {
        if ( ConstInfo().GetKind() != KIND_ENUM )
        {
            return (std::string)(*this);
        }

        std::string enumString( ConstInfo().GetEnum() );

        std::vector<std::string> output;

        std::istringstream is( enumString );
        std::string part;
        while ( getline(is, part, ',') )
        {
            part.erase(std::remove(part.begin(), part.end(), ' '), part.end());

            output.push_back( part );
        }

        unsigned int enumPos = (int)(*this);

        if ( enumPos >= output.size() )
        {
            return "unknown";
        }

        return output[ enumPos ];
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int PropertyBase::ToEnumPosition( const std::string& enumStringValue ) const
    {
        if ( ConstInfo().GetKind() != KIND_ENUM )
        {
            return 0;
        }

        std::string enumString( ConstInfo().GetEnum() );

        int pos = 0;

        std::istringstream is( enumString );
        std::string part;
        while ( getline(is, part, ',') )
        {
            part.erase(std::remove(part.begin(), part.end(), ' '), part.end());

            if ( part == enumStringValue )
            {
                return pos;
            }
            pos++;
        }

        return pos;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool_t PropertyBase::IsReference() const
    {
        if ( cInstanceManager::IsInstance( dynamic_cast<cInstanceManager*>(smart_ptr_getRaw(m_referenceParent)) ) )
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
    PropertyBase::operator bool() const
    {
        return 0;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase::operator char() const
    {
        return 0;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase::operator unsigned char() const
    {
        return 0;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase::operator int() const
    {
        return 0;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase::operator unsigned int() const
    {
        return 0;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase::operator unsigned short() const
    {
        return 0;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase::operator double() const
    {
        return 0;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase::operator float() const
    {
        return 0;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase::operator std::string() const
    {
        return "";
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertyBase::EmitChanges()
    {
        signalChanged.Emit( this );
    }
}
