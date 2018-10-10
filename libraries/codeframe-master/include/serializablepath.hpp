#ifndef SERIALIZABLEPATH_HPP_INCLUDED
#define SERIALIZABLEPATH_HPP_INCLUDED

#include <string>

namespace codeframe
{
    class cSerializableInterface;

    class cSerializablePath
    {
        public:
             cSerializablePath( cSerializableInterface& sint );
            ~cSerializablePath();

            std::string PathString() const;

        private:
            cSerializableInterface& m_sint;
    };

}

#endif // SERIALIZABLEPATH_HPP_INCLUDED
