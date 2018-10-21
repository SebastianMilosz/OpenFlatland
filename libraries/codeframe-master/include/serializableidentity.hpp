#ifndef SERIALIZABLEIDENTITY_HPP_INCLUDED
#define SERIALIZABLEIDENTITY_HPP_INCLUDED

#include <string>

namespace codeframe
{
    class cSerializableIdentity
    {
        public:
             cSerializableIdentity( const std::string& name );
            ~cSerializableIdentity();

            void SetName ( const std::string& name );
            std::string ObjectName( bool idSuffix = true ) const;

            int  GetId() const { return m_Id; }
            void SetId( int id ) { m_Id = id; }

            void    EnterPulseState()    { m_pulseState = true;  }
            void    LeavePulseState()    { m_pulseState = false; }
            bool    IsPulseState() const { return  m_pulseState; }

            // Library version nr. and string
            static float       LibraryVersion();
            static std::string LibraryVersionString();
        private:
            int m_Id;
            std::string m_sContainerName;
            bool m_pulseState;
    };
}

#endif // SERIALIZABLEIDENTITY_HPP_INCLUDED
