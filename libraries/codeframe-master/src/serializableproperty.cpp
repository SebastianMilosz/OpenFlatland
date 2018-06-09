#include <exception>
#include <stdexcept>
#include <sstream>
#include <algorithm>

#include "serializableproperty.h"
#include "instancemanager.h"
#include "serializable.h"

namespace codeframe
{

    int Property::s_globalParConCnt = 0;

    /*****************************************************************************/
    /**
      * @brief Zamienia napis na liczbe 32b
     **
    ******************************************************************************/
    uint32_t Property::GetHashId( std::string str, uint16_t mod )
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
    void Property::RegisterProperty()
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
    void Property::UnRegisterProperty()
    {
        if( m_parentpc == NULL ) return;

        m_Mutex.Lock();

        m_parentpc->UnRegisterProperty( this );

        s_globalParConCnt--;

        m_Mutex.Unlock();
    }

    /*****************************************************************************/
    /**
      * @brief Operator porownania
     **
    ******************************************************************************/
    bool Property::operator==(const Property& sval)
    {
        m_Mutex.Lock();
        bool retVal = false;
        if(m_type == sval.m_type)
        {
            switch ( (int)m_type )
            {
              case TYPE_TEXT:         { if(type_text   == sval.type_text  ) { retVal = true; } break; }
              case TYPE_INT:          { if(v.type_int  == sval.v.type_int ) { retVal = true; } break; }
              case TYPE_REAL:         { if(v.type_real == sval.v.type_real) { retVal = true; } break; }
              case TYPE_CHAR:         { if(v.type_char == sval.v.type_char) { retVal = true; } break; }
            }
        }
        m_Mutex.Unlock();

        return retVal;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool Property::operator!=(const Property& sval)
    {
        m_Mutex.Lock();
        bool retVal = false;
        if(m_type != sval.m_type)
        {
            switch( (int)m_type )
            {
              case TYPE_TEXT:         { if(type_text   != sval.type_text  ) { retVal = true; } break; }
              case TYPE_INT:          { if(v.type_int  != sval.v.type_int ) { retVal = true; } break; }
              case TYPE_REAL:         { if(v.type_real != sval.v.type_real) { retVal = true; } break; }
              case TYPE_CHAR:         { if(v.type_char != sval.v.type_char) { retVal = true; } break; }
            }
        }
        m_Mutex.Unlock();

        return retVal;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool Property::operator==(const int& sval)
    {
        m_Mutex.Lock();

        bool retVal = false;
        switch ( (int)m_type )
        {
          case TYPE_INT:          { if((int)v.type_int  == sval ) { retVal = true; } break; }
          case TYPE_REAL:         { if((int)v.type_real == sval ) { retVal = true; } break; }
          case TYPE_CHAR:         { if((int)v.type_char == sval ) { retVal = true; } break; }
        }

        m_Mutex.Unlock();

        return retVal;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool Property::operator!=(const int& sval)
    {
        m_Mutex.Lock();
        bool retVal = false;
        switch ( (int)m_type )
        {
          case TYPE_INT:          { if((int)v.type_int  != sval ) { retVal = true; } break; }
          case TYPE_REAL:         { if((int)v.type_real != sval ) { retVal = true; } break; }
          case TYPE_CHAR:         { if((int)v.type_char != sval ) { retVal = true; } break; }
        }
        m_Mutex.Unlock();

        return retVal;
    }

    /*****************************************************************************/
    /**
      * @brief Operatory przypisania
     **
    ******************************************************************************/
    Property& Property::operator=(Property val)
    {
        m_Mutex.Lock();
        val.m_Mutex.Lock();

        m_reference       = val.m_reference;
        m_name            = val.m_name;
        m_id              = val.m_id;
        m_type            = val.m_type;
        v                 = val.v;
        type_text         = val.type_text;
        m_parentpc        = val.m_parentpc;

        // Wartosc historyczna
        prew_type_text    = val.prew_type_text;
        prew_v            = val.prew_v;

        m_Mutex.Unlock();
        val.m_Mutex.Unlock();

        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    Property& Property::operator=(bool val)
    {
        if( Info().GetEnable() == true )
        {
            m_Mutex.Lock();

            if( v.type_char != val )
            {
                prew_v.type_char = v.type_char;
                v.type_char      = val;
                m_type           = TYPE_CHAR;

                if(m_propertyInfo.IsEventEnable()) { signalChanged.Emit( this ); }
            }

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
    Property& Property::operator=(char val)
    {
        if( Info().GetEnable() == true )
        {
            m_Mutex.Lock();

            if( v.type_char != val )
            {
                prew_v.type_char = v.type_char;
                m_type           = TYPE_CHAR;
                v.type_char      = val;

                if(m_propertyInfo.IsEventEnable()) { signalChanged.Emit( this ); }
            }

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
    Property& Property::operator=(int val)
    {
        if( Info().GetEnable() == true )
        {
            m_Mutex.Lock();

            if( v.type_int != val )
            {
                prew_v.type_int = v.type_int;
                m_type          = TYPE_INT;
                v.type_int      = val;

                if(m_propertyInfo.IsEventEnable()) { signalChanged.Emit( this ); }
            }

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
    Property& Property::operator=(unsigned int val)
    {
        if( Info().GetEnable() == true )
        {
            m_Mutex.Lock();

            if( v.type_int != (int)val )
            {
                prew_v.type_int = v.type_int;
                m_type          = TYPE_INT;
                v.type_int      = val;

                if(m_propertyInfo.IsEventEnable()) { signalChanged.Emit( this ); }
            }

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
    Property& Property::operator=(double val)
    {
        if( Info().GetEnable() == true )
        {
            m_Mutex.Lock();

            if(v.type_real != val)
            {
                prew_v.type_real = v.type_real;
                m_type           = TYPE_REAL;
                v.type_real      = val;

                if(m_propertyInfo.IsEventEnable()) { signalChanged.Emit( this ); }
            }

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
    Property& Property::operator=(std::string val)
    {
        if( Info().GetEnable() == true )
        {
            m_Mutex.Lock();

            if( type_text != val )
            {
                prew_type_text = type_text;
                m_type         = TYPE_TEXT;
                type_text      = val;

                if(m_propertyInfo.IsEventEnable()) { signalChanged.Emit( this ); }
            }

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
    Property& Property::operator++()
    {
        (*this) = (int)(*this) + 1;

        // actual increment takes place here
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    Property Property::operator++(int)
    {
        Property tmp(*this);    // copy
        operator++();           // pre-increment
        return tmp;             // return old value
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    Property& Property::operator--()
    {
        (*this) = (int)(*this) - 1;

        // actual decrement takes place here
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    Property Property::operator--(int)
    {
        Property tmp(*this);    // copy
        operator--();           // pre-decrement
        return tmp;             // return old value
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    Property& Property::operator+=(const Property& rhs)
    {
        *this = *this + rhs;
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    Property& Property::operator-=(const Property& rhs)
    {
        *this = *this - rhs;
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    Property Property::operator+(const Property& rhs)
    {
        Property prop(*this);

        m_Mutex.Lock();
        switch ( (int)m_type )
        {
          case TYPE_INT:  { prop = (int)*this    + (int)rhs; break; }
          case TYPE_REAL: { prop = (double)*this + (double)rhs; break; }
          case TYPE_CHAR: { prop = (char)*this   + (char)rhs; break; }
        }
        m_Mutex.Unlock();

        return prop;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    Property Property::operator-(const Property& rhs)
    {
        Property prop(*this);

        m_Mutex.Lock();
        switch ( (int)m_type )
        {
          case TYPE_INT:  { prop = (int)*this    - (int)rhs;    break; }
          case TYPE_REAL: { prop = (double)*this - (double)rhs; break; }
          case TYPE_CHAR: { prop = (char)*this   - (char)rhs;   break; }
        }
        m_Mutex.Unlock();

        return prop;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    Property& Property::operator+=(const int rhs)
    {
        Property prop(*this);
        prop = rhs;

        *this = *this + prop;
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    Property& Property::operator-=(const int rhs)
    {
        Property prop(*this);
        prop = rhs;

        *this = *this - prop;
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    Property::operator bool() const
    {
       if( m_reference ) { return (bool)(*m_reference); }

       return (bool)((char)(*this));
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    Property::operator char() const
    {
        if( m_reference ) { return (char)(*m_reference); }

        if(m_type == TYPE_CHAR) { return v.type_char; }
        else return (char)((int)(*this));
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    Property::operator int() const
    {
        if( m_reference ) { return (int)(*m_reference); }

             if(m_type == TYPE_INT)  { return v.type_int;  }
        else if(m_type == TYPE_CHAR) { return v.type_char; }
        else return (int)((double)(*this));
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    Property::operator unsigned int() const
    {
        if( m_reference ) { return (int)(*m_reference); }

             if(m_type == TYPE_INT)  { return v.type_int;  }
        else if(m_type == TYPE_CHAR) { return v.type_char; }
        else return (unsigned int)((double)(*this));
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    Property::operator unsigned short() const
    {
        if( m_reference ) { return (unsigned short)(*m_reference); }

        return (unsigned short)((int)(*this));
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    Property::operator double() const
    {
        if( m_reference ) { return (double)(*m_reference); }

             if(m_type == TYPE_REAL) { return v.type_real; }
        else if(m_type == TYPE_INT)  { return v.type_int;  }
        else if(m_type == TYPE_CHAR) { return v.type_char; }
        else return 0;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    Property::operator std::string() const
    {
        if( m_reference ) { return (std::string)(*m_reference); }

             if(m_type == TYPE_TEXT) { return type_text;                                }
        else if(m_type == TYPE_REAL) { return utilities::math::FloatToStr(v.type_real); }
        else if(m_type == TYPE_INT)  { return utilities::math::IntToStr(v.type_int);    }
        else if(m_type == TYPE_CHAR) { return utilities::math::IntToStr(v.type_char);   }
        else return "";
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string Property::Name() const
    {
        return m_name;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool Property::NameIs( std::string name ) const
    {
        if( name == m_name ) return true;
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    uint32_t Property::Id() const
    {
        return m_id;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    eType Property::Type() const
    {
        return m_type;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string Property::PreviousValueString() const
    {
        switch ( m_type )
        {
            case TYPE_TEXT: { return prew_type_text;                                }
            case TYPE_INT:  { return utilities::math::IntToStr(prew_v.type_int);    }
            case TYPE_REAL: { return utilities::math::FloatToStr(prew_v.type_real); }
            case TYPE_CHAR: { return utilities::math::IntToStr(prew_v.type_char);   }
            default:        { return "unknown";                                     }
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string Property::CurentValueString() const
    {
        switch ( m_type )
        {
            case TYPE_TEXT: { return type_text;                                }
            case TYPE_INT:  { return utilities::math::IntToStr(v.type_int);    }
            case TYPE_REAL: { return utilities::math::FloatToStr(v.type_real); }
            case TYPE_CHAR: { return utilities::math::IntToStr(v.type_char);   }
            default:        { return "unknown";                                }
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int Property::PreviousValueInteger() const
    {
        switch ( m_type )
        {
            case TYPE_TEXT: { return 0;                }
            case TYPE_INT:  { return prew_v.type_int;  }
            case TYPE_REAL: { return prew_v.type_real; }
            case TYPE_CHAR: { return prew_v.type_char; }
            default:        { return 0;                }
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int Property::CurentValueInteger() const
    {
        switch ( m_type )
        {
            case TYPE_TEXT: { return 0;           }
            case TYPE_INT:  { return v.type_int;  }
            case TYPE_REAL: { return v.type_real; }
            case TYPE_CHAR: { return v.type_char; }
            default:        { return 0;           }
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string Property::TypeString() const
    {
        switch ( m_type )
        {
            case TYPE_TEXT:         { return "text";          }
            case TYPE_INT:          { return "int";           }
            case TYPE_REAL:         { return "real";          }
            case TYPE_CHAR:         { return "char";          }
            default:                { return "default";       }
        }
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string Property::Path(bool addName) const
    {
        std::string propPath;

        if( m_parentpc )
        {
            propPath = m_parentpc->Path();
            if(addName) { propPath += "." + Name(); }
        }
        else
        {
            throw std::runtime_error( "Property::Path() NULL Parent Container" );
        }

        return propPath;
    }

    /*****************************************************************************/
    /**
      * @brief Wymusza na zmiennej ze ma czekac na aktualizacje w wypadku
      * jej braku jest generowany event
     **
    ******************************************************************************/
    void Property::WaitForUpdate( int time )
    {
        m_isWaitForUpdate  = true;
        m_waitForUpdateCnt = time;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void Property::ValueUpdate()
    {
        m_isWaitForUpdate  = false;
        m_waitForUpdateCnt = 0;
    }

    /*****************************************************************************/
    /**
      * @brief Wyzwalacz licznika dla oczekiwania aktualizacji rejestru
     **
    ******************************************************************************/
    void Property::WaitForUpdatePulse()
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
    bool Property::ConnectReference( Property* refProp )
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
    void Property::CommitChanges()
    {
        if( m_parentpc &&  m_parentpc->IsPulseState() )
        {
            m_pulseAbort = true;
        }
        prew_type_text = type_text;
        prew_v         = v;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool Property::IsChanged() const
    {
        if( prew_type_text   != type_text  ||
            prew_v.type_int  != v.type_int ||
            prew_v.type_char != v.type_char ) return true;
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void Property::PulseChanged()
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
    Property& Property::WatchdogGetValue( int time )
    {
        WaitForUpdate( time );
        return *this;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void Property::SetNumber( int val )
    {
        if(Info().GetEnable() == false) return;

        *this = (int)val;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    int Property::GetNumber() const
    {
        Property prop = *this;
        return (int)prop;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void Property::SetReal( double val )
    {
        if(Info().GetEnable() == false) return;

        *this = (double)val;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    double Property::GetReal() const
    {
        Property prop = *this;
        return (double)prop;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void Property::SetString( std::string  val )
    {
        if(Info().GetEnable() == false) return;

        *this = (std::string)val;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string Property::GetString() const
    {
        Property prop = *this;
        return (std::string)prop;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    std::string Property::ToString()
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
    int Property::ToEnumPosition( std::string enumStringValue )
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
    bool Property::IsReference() const
    {
        if( cInstanceManager::IsInstance( dynamic_cast<cInstanceManager*>(m_referenceParent) ) ) { return true; }
        return false;
    }

}
