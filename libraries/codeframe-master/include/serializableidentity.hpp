#ifndef SERIALIZABLEIDENTITY_HPP_INCLUDED
#define SERIALIZABLEIDENTITY_HPP_INCLUDED

#include <string>

namespace codeframe
{
    class cSerializableInterface;

    enum eBuildType
    {
        STATIC,
        DYNAMIC
    };

    enum eBuildRole
    {
        OBJECT,
        CONTAINER
    };

    class cSerializableIdentity
    {
        public:
             cSerializableIdentity( const std::string& name, cSerializableInterface& sint );
            ~cSerializableIdentity();

            void SetName ( const std::string& name );
            std::string ObjectName( bool idSuffix = true ) const;

            uint32_t GetUId() const { return m_uid; }
            int32_t  GetId()  const { return m_id; }
            void SetId( const int32_t id ) { m_id = id; }

            void    EnterPulseState()    { m_pulseState = true;  }
            void    LeavePulseState()    { m_pulseState = false; }
            bool    IsPulseState() const { return  m_pulseState; }

            // Library version nr. and string
            static float       LibraryVersion();
            static std::string LibraryVersionString();
        private:
             int32_t m_id;
            uint32_t m_uid;
            static uint32_t g_uid;
            cSerializableInterface& m_sint;
            std::string m_sContainerName;
            bool m_pulseState;
    };
}

#endif // SERIALIZABLEIDENTITY_HPP_INCLUDED
