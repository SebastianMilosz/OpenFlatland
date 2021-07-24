#ifndef THREADWRAPPER_H
#define THREADWRAPPER_H

#define __CLEANUP_CXX
#define HAVE_STRUCT_TIMESPEC            // redefinition of struct timespec

#include <pthread.h>
#include <unistd.h>
#include <errno.h>
#include <string>
#include <sigslot.h>

using namespace sigslot;

enum { WRTHREAD_MIN_PRIORITY = 0, WRTHREAD_NORMAL_PRIORITY, WRTHREAD_HIGH_PRIORITY };

/*****************************************************************************/
/**
  * @brief Threads wrapper - mutexes
 **
******************************************************************************/
class WrMutex
{
    public:
        friend class WrThreadEvent;

        WrMutex();
       ~WrMutex();
        WrMutex(const WrMutex& copy);

        bool Lock() const;
        bool Unlock() const;
        bool TryLock() const;
        bool TryLock(unsigned long ms) const;

    private:
        mutable pthread_mutex_t *m_pmutex;
        bool                     m_isOwner;
};

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
class cMutexLocker
{
private:
    WrMutex* m_mutex;

public:
    cMutexLocker( WrMutex* mutex );
   ~cMutexLocker();
};

/*****************************************************************************/
/**
  * @brief Threads wrapper - events
 **
******************************************************************************/
class WrThreadEvent
{
    public:
        WrThreadEvent();
        ~WrThreadEvent();
        WrThreadEvent(const WrThreadEvent& copy);

        void   Notify();
        void   NotifyAll();
        void   Wait(WrMutex& mutex);
        bool   Wait(WrMutex& mutex, long msec);

    private:
        pthread_cond_t      *m_pcond;
        bool                 m_isOwner;
};

/*****************************************************************************/
/**
  * @brief Threads wrapper
 **
******************************************************************************/
class WrThread
{
    public:
                 WrThread();
        virtual ~WrThread();

        // Sterowanie watkiem wbudowanym
        void   Start();
        void   Stop();
        void   Pause();
        void   Join();
        void   Exit();
        bool   IsRunning();
        int    GetPriority();
        void   SetPriority( int priority );
        void   Sleep( unsigned long msec );

        void TerminateProcessing();

        static uint8_t NCPU();

        enum eTState {
            IDLE         = 0x00,
            PAUSE        = 0x01,
            PROCESSING   = 0x02,
            CLOSING      = 0x03,
            REDYTODELETE = 0x04,
            SHUTDOWN     = 0x05
            };

    protected:
        virtual bool ForceTerminateProcessing(); ///< Metoda powinna zawierac kod wymuszajacy zamkniecie watku tj. zakonczenie procesow blokujacych
        virtual bool StartProcessing() = 0;
        virtual bool Processing() = 0;
        virtual bool EndProcessing() = 0;

        volatile eTState m_ThreadState;

    private:
        pthread_t      m_thread;
        pthread_attr_t m_thread_attr;
        volatile bool  m_running;

        void Exec();
        static void* ExecFunc(void* p_this);
};

/*****************************************************************************/
/**
  * @brief Threads wrapper
 **
******************************************************************************/
class WrTimer : public WrThread
{
private:
    int m_timeout;

    bool StartProcessing();
    bool Processing();
    bool EndProcessing();

public:
    WrTimer( int timeout );
   ~WrTimer();

   signal0<> signalTimeout;
};

#endif // THREADWRAPPER_H
