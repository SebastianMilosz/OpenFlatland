#ifndef CSERIALIZABLESTREAMMANAGER_H
#define CSERIALIZABLESTREAMMANAGER_H

#include <serializablecontainer.h>
#include <serializablestream.h>

namespace codeframe
{

    class cSerializableStreamManager : public cSerializableContainer< cSerializableStream >
    {
        public:
            std::string Role()      { return "Container"; }
            std::string Class()     { return "cSerializableStreamManager"; }

        public:
                     cSerializableStreamManager( cSerializable* parentObject );
            virtual ~cSerializableStreamManager();

            Property_Int Enable;

            cSerializableStream* Create( std::string className, std::string objName, int cnt = -1 );

        private:
            void OnEnable( Property* prop );
    };

}

#endif // CSERIALIZABLESTREAMMANAGER_H
