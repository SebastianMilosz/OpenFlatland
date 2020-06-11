#ifndef INSTANCE_MANAGER_H_INCLUDED
#define INSTANCE_MANAGER_H_INCLUDED

#include <vector>
#include <algorithm>
#include <ThreadUtilities.h>

namespace codeframe
{

    /*****************************************************************************/
    /**
      * @brief
      * @version 1.0
     **
    ******************************************************************************/
    class cInstanceManager
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

#endif // INSTANCE_MANAGER_H_INCLUDED
