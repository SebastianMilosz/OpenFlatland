#ifndef INSTANCEMANAGER_H_INCLUDED
#define INSTANCEMANAGER_H_INCLUDED

#include "serializableinterface.h"

#include <vector>
#include <algorithm>
#include <ThreadUtilities.h>

namespace codeframe
{

    /*****************************************************************************/
    /**
      * @brief
      * @author Sebastian Milosz
      * @version 1.0
     **
    ******************************************************************************/
    class cInstanceManager : public cSerializableInterface
    {
        public:
                 cInstanceManager();
        virtual ~cInstanceManager();

        static bool IsInstance( void* ptr );
        static void DestructorEnter( void* ptr );
        static int  GetInstanceCnt();

    private:
        static std::vector<void*>  s_vInstanceList;
        static WrMutex             s_Mutex;
        static int                 s_InstanceCnt;
    };

}

#endif // INSTANCEMANAGER_H_INCLUDED
