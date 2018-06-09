#ifndef CSERIALIZABLESTORAGE_H
#define CSERIALIZABLESTORAGE_H

#include "serializableinterface.h"
#include "instancemanager.h"

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief This class add storage functionality to cInstanceManager
      * @author Sebastian Milosz
      * @version 1.0
      * @note cSetializable
     **
    ******************************************************************************/
    class cSerializableStorage : public cInstanceManager
    {
        public:
                     cSerializableStorage();
            virtual ~cSerializableStorage();

            enum eShareLevel { ShareThis = 0, ShareFull };

        protected:
            eShareLevel m_shareLevel;
    };

}

#endif // CSERIALIZABLESTORAGE_H
