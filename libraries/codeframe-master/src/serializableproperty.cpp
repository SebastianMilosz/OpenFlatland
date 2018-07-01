#include <exception>
#include <stdexcept>
#include <sstream>
#include <algorithm>

#include "serializableproperty.h"
#include "instancemanager.h"
#include "serializable.h"

namespace codeframe
{

    int PropertyBase::s_globalParConCnt = 0;

    /*****************************************************************************/
    /**
      * @brief Zamienia napis na liczbe 32b
     **
    ******************************************************************************/
    uint32_t PropertyBase::GetHashId( std::string str, uint16_t mod )
    {
        uint32_t hash, i;
        for(hash = i = 0; i < str.size(); ++i)
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
    void PropertyBase::RegisterProperty()
    {
        if( m_parentpc == NULL ) return;

        m_Mutex.Lock();

        int size = m_parentpc->GetObjectFieldCnt();

        m_id = GetHashId( Name(), 255 * s_globalParConCnt + size );

        m_parentpc->RegisterProperty( this );

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
        if( m_parentpc == NULL ) return;

        m_Mutex.Lock();

        m_parentpc->UnRegisterProperty( this );

        s_globalParConCnt--;

        m_Mutex.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief Operatory przypisania
     **
    ******************************************************************************/
    PropertyBase& PropertyBase::operator=(PropertyBase val)
    {
        m_Mutex.Lock();
        val.m_Mutex.Lock();

        m_reference       = val.m_reference;
        m_name            = val.m_name;
        m_id              = val.m_id;
        m_type            = val.m_type;
        m_parentpc        = val.m_parentpc;

        m_Mutex.Unlock();
        val.m_Mutex.Unlock();

        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase& PropertyBase::operator=(bool val)
    {
        if( Info().GetEnable() == true )
        {
            m_Mutex.Lock();

            //if( v.type_char != val )
            //{
            //    prew_v.type_char = v.type_char;
            //    v.type_char      = val;
            //    m_type           = TYPE_CHAR;

                if(m_propertyInfo.IsEventEnable()) { signalChanged.Emit( this ); }
            //}

            // Przypisanie wartosci zdalnej referencji
            if( m_reference )
            {
                *m_reference = val;
            }

            ValueUpdate();

            m_Mutex.Unlock();
        }

        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase& PropertyBase::operator=(char val)
    {
        if( Info().GetEnable() == true )
        {
            m_Mutex.Lock();

            //if( v.type_char != val )
            //{
            //    prew_v.type_char = v.type_char;
            //    m_type           = TYPE_CHAR;
            //    v.type_char      = val;

                if(m_propertyInfo.IsEventEnable()) { signalChanged.Emit( this ); }
            //}

            // Przypisanie wartosci zdalnej referencji
            if( m_reference )
            {
                *m_reference = val;
            }

            ValueUpdate();

            m_Mutex.Unlock();
        }

        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase& PropertyBase::operator=(int val)
    {
        if( Info().GetEnable() == true )
        {
            m_Mutex.Lock();

            //if( v.type_int != val )
            //{
            //    prew_v.type_int = v.type_int;
            //    m_type          = TYPE_INT;
            //    v.type_int      = val;

                if(m_propertyInfo.IsEventEnable()) { signalChanged.Emit( this ); }
            //}

            // Przypisanie wartosci zdalnej referencji
            if( m_reference )
            {
                *m_reference = val;
            }

            ValueUpdate();

            m_Mutex.Unlock();
        }

        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase& PropertyBase::operator=(unsigned int val)
    {
        if( Info().GetEnable() == true )
        {
            m_Mutex.Lock();

            //if( v.type_int != (int)val )
            //{
            //    prew_v.type_int = v.type_int;
            //    m_type          = TYPE_INT;
            //    v.type_int      = val;

                if(m_propertyInfo.IsEventEnable()) { signalChanged.Emit( this ); }
            //}

            // Przypisanie wartosci zdalnej referencji
            if( m_reference )
            {
                *m_reference = val;
            }

            ValueUpdate();

            m_Mutex.Unlock();
        }

        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase& PropertyBase::operator=(double val)
    {
        if( Info().GetEnable() == true )
        {
            m_Mutex.Lock();

            //if(v.type_real != val)
            //{
            //    prew_v.type_real = v.type_real;
            //    m_type           = TYPE_REAL;
            //    v.type_real      = val;

                if(m_propertyInfo.IsEventEnable()) { signalChanged.Emit( this ); }
            //}

            // Przypisanie wartosci zdalnej referencji
            if( m_reference )
            {
                *m_reference = val;
            }

            ValueUpdate();

            m_Mutex.Unlock();
        }

        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase& PropertyBase::operator=(std::string val)
    {
        if( Info().GetEnable() == true )
        {
            m_Mutex.Lock();

            //if( type_text != val )
            //{
            //    prew_type_text = type_text;
            //    m_type         = TYPE_TEXT;
            //    type_text      = val;

                if(m_propertyInfo.IsEventEnable()) { signalChanged.Emit( this ); }
            //}

            // Przypisanie wartosci zdalnej referencji
            if( m_reference )
            {
                *m_reference = val;
            }

            ValueUpdate();

            m_Mutex.Unlock();
        }

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
    bool PropertyBase::NameIs( std::string name ) const
    {
        if( name == m_name ) return true;
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
        switch ( m_type )
        {
            case TYPE_TEXT:
            {
                //return prew_type_text;
            }
            case TYPE_INT:
            {
                //return utilities::math::IntToStr(prew_v.type_int);
            }
            case TYPE_REAL:
            {
                //return utilities::math::FloatToStr(prew_v.type_real);
            }
            case TYPE_CHAR:
            {
                //return utilities::math::IntToStr(prew_v.type_char);
            }
            default:
            {
                return "unknown";
            }
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertyBase::CurentValueString() const
    {
        switch ( m_type )
        {
            //case TYPE_TEXT: { return type_text;                                }
            //case TYPE_INT:  { return utilities::math::IntToStr(v.type_int);    }
            //case TYPE_REAL: { return utilities::math::FloatToStr(v.type_real); }
            //case TYPE_CHAR: { return utilities::math::IntToStr(v.type_char);   }
            default:        { return "unknown";                                }
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int PropertyBase::PreviousValueInteger() const
    {
        switch ( m_type )
        {
            //case TYPE_TEXT: { return 0;                }
            //case TYPE_INT:  { return prew_v.type_int;  }
            //case TYPE_REAL: { return prew_v.type_real; }
            //case TYPE_CHAR: { return prew_v.type_char; }
            default:        { return 0;                }
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int PropertyBase::CurentValueInteger() const
    {
        switch ( m_type )
        {
            //case TYPE_TEXT: { return 0;           }
            //case TYPE_INT:  { return v.type_int;  }
            //case TYPE_REAL: { return v.type_real; }
            //case TYPE_CHAR: { return v.type_char; }
            default:        { return 0;           }
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertyBase::TypeString() const
    {
        switch ( m_type )
        {
            //case TYPE_TEXT:         { return "text";          }
            //case TYPE_INT:          { return "int";           }
            //case TYPE_REAL:         { return "real";          }
            //case TYPE_CHAR:         { return "char";          }
            default:                { return "default";       }
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string PropertyBase::Path(bool addName) const
    {
        std::string propPath;

        if( m_parentpc )
        {
            propPath = m_parentpc->Path();
            if(addName) { propPath += "." + Name(); }
        }
        else
        {
            throw std::runtime_error( "PropertyBase::Path() NULL Parent Container" );
        }

        return propPath;
    }

    /*****************************************************************************/
    /**
      * @brief Wymusza na zmiennej ze ma czekac na aktualizacje w wypadku
      * jej braku jest generowany event
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
    void PropertyBase::ValueUpdate()
    {
        m_isWaitForUpdate  = false;
        m_waitForUpdateCnt = 0;
    }

    /*****************************************************************************/
    /**
      * @brief Wyzwalacz licznika dla oczekiwania aktualizacji rejestru
     **
    ******************************************************************************/
    void PropertyBase::WaitForUpdatePulse()
    {
        if(m_isWaitForUpdate == true && m_waitForUpdateCnt > 0)
        {
           m_waitForUpdateCnt--;

           if(m_waitForUpdateCnt == 0)
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
    bool PropertyBase::ConnectReference( PropertyBase* refProp )
    {
        // Sprawdzamy czy zgadza sie typ
        if( refProp != NULL && this->Type() == refProp->Type() )
        {
            m_Mutex.Lock();
            m_referenceParent   = refProp->m_parentpc;
            m_reference         = refProp;
            m_Mutex.Unlock();
            return true;
        }

        return false;
    }

    /*****************************************************************************/
    /**
      * @brief Zatwierdzenie wszystkich zmian obiektu i jego potomnych
     **
    ******************************************************************************/
    void PropertyBase::CommitChanges()
    {
        if( m_parentpc &&  m_parentpc->IsPulseState() )
        {
            m_pulseAbort = true;
        }
        //prew_type_text = type_text;
        //prew_v         = v;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertyBase::IsChanged() const
    {
        /*
        if( prew_type_text   != type_text  ||
            prew_v.type_int  != v.type_int ||
            prew_v.type_char != v.type_char ) return true;
            */
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertyBase::PulseChanged()
    {
        if( m_pulseAbort && m_parentpc &&  m_parentpc->IsPulseState() ) { return; }

        m_pulseAbort = false;

        signalChanged.Emit( this );
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    PropertyBase& PropertyBase::WatchdogGetValue( int time )
    {
        WaitForUpdate( time );
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void PropertyBase::SetNumber( int val )
    {
        if(Info().GetEnable() == false) return;

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
    void PropertyBase::SetReal( double val )
    {
        if(Info().GetEnable() == false) return;

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
    void PropertyBase::SetString( std::string  val )
    {
        if(Info().GetEnable() == false) return;

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
    std::string PropertyBase::ToString()
    {
        if( Info().GetKind() != KIND_ENUM ) return (std::string)(*this);

        std::string enumString = Info().GetEnum();

        std::vector<std::string> output;

        std::istringstream is( enumString );
        std::string part;
        while (getline(is, part, ','))
        {
            part.erase(std::remove(part.begin(), part.end(), ' '), part.end());

            output.push_back( part );
        }

        unsigned int enumPos = (int)(*this);

        if( enumPos >= output.size() ) return "unknown";

        return output[ enumPos ];
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int PropertyBase::ToEnumPosition( std::string enumStringValue )
    {
        if( Info().GetKind() != KIND_ENUM ) return 0;

        std::string enumString = Info().GetEnum();

        int pos = 0;

        std::istringstream is( enumString );
        std::string part;
        while (getline(is, part, ','))
        {
            part.erase(std::remove(part.begin(), part.end(), ' '), part.end());

            if( part == enumStringValue ) return pos;
            pos++;
        }

        return pos;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool PropertyBase::IsReference() const
    {
        if( cInstanceManager::IsInstance( dynamic_cast<cInstanceManager*>(m_referenceParent) ) ) { return true; }
        return false;
    }

}
