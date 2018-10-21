#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <utilities/ThreadUtilities.h>
#include <utilities/DataTypesUtilities.h>

#ifdef WIN32
#include <windows.h>
#endif

/*****************************************************************************/
/**
  * @brief  Return number of CPU cores
  * @return number of CPU cores
 **
******************************************************************************/
uint8_t WrThread::NCPU()
{
#ifdef WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;;
#else
    return 1;
#endif
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
WrThread::WrThread() : m_running(false)
{
    m_ThreadState = IDLE;
    pthread_attr_init( &m_thread_attr );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
WrThread::~WrThread()
{
    m_ThreadState = SHUTDOWN;
    Join();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool WrThread::ForceTerminateProcessing()
{
    return true;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void WrThread::TerminateProcessing()
{
    if ( m_running )
    {
        if ( m_ThreadState < CLOSING )
        {
            m_ThreadState = CLOSING;
            ForceTerminateProcessing();

            // Oczekiwanie na stan gotowy do zamkniecia
            while (m_ThreadState < REDYTODELETE)
            {
            }
        }
        else
        {
            m_ThreadState = REDYTODELETE;
        }
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void WrThread::Exec()
{
    StartProcessing();

    if (m_ThreadState < PROCESSING) m_ThreadState = PROCESSING;

    while ( m_ThreadState == PROCESSING || m_ThreadState == PAUSE )
    {
        if (m_ThreadState == PAUSE) { Sleep( 500U ); continue; }

        if ( Processing() == false ) { m_ThreadState = REDYTODELETE; break; }
    }

    EndProcessing();

    if (m_ThreadState < REDYTODELETE) m_ThreadState = REDYTODELETE;

    while ( m_ThreadState <= REDYTODELETE)
    {
        Sleep( 500U );
    }
}

/*****************************************************************************/
/**
  * @brief
  * @param p_this Pointer to WrThread
 **
******************************************************************************/
void* WrThread::ExecFunc(void* p_this)
{
    WrThread * tp_this = static_cast<WrThread*>(p_this);
    tp_this->Exec();
    tp_this->m_running = false;

    return NULL;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void WrThread::Start()
{
    if ( m_running == true )
    {
        return;
    }

    if ( pthread_create(&m_thread, &m_thread_attr, ExecFunc, static_cast<void *>(this)) == 0U )
    {
        m_running = true;
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void WrThread::Stop()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void WrThread::Pause()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool WrThread::IsRunning()
{
    return m_running;
}

/*****************************************************************************/
/**
  * @brief pthreads-win32 support only SCHED_OTHER and priority is always 0
  * @return pthreads-win32 always 0
 **
******************************************************************************/
int WrThread::GetPriority()
{
    return 0U;
}

/*****************************************************************************/
/**
  * @brief pthreads-win32 support only SCHED_OTHER and priority is always 0
 **
******************************************************************************/
void WrThread::SetPriority( int priority )
{
    (void)priority;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void WrThread::Exit()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void WrThread::Join()
{
    while ( m_running == true ) {}

    //void *retValue;
    //m_running = false;

    //pthread_join(m_thread, &retValue);
}

/*****************************************************************************/
/**
  * @brief
  * @param msec The time interval for which execution is to be suspended, in
  *				milliseconds
 **
******************************************************************************/
void WrThread::Sleep(unsigned long msec)
{
#ifdef WIN32
    ::Sleep(msec);
#else // _WIN32
    sleep(msec);
#endif
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
WrMutex::WrMutex() :
       m_pmutex( 0U ),
       m_isOwner( true )
{
    m_pmutex = new pthread_mutex_t;
    if ( pthread_mutex_init(m_pmutex, 0U) == 0U )
    {
        //ok
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
WrMutex::~WrMutex()
{
    if ( m_isOwner == true )
    {
        if ( pthread_mutex_destroy( m_pmutex ) == EBUSY )
        {
            Unlock();
            pthread_mutex_destroy( m_pmutex );
        }
        delete m_pmutex;
    }
}

/*****************************************************************************/
/**
  * @brief
  * @param copy Reference to original WrMutex
 **
******************************************************************************/
WrMutex::WrMutex( const WrMutex& copy )
{
    m_pmutex = copy.m_pmutex;
    m_isOwner = false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool WrMutex::Lock() const
{
    if ( pthread_mutex_lock( m_pmutex ) == 0U )
    {
        return true;
    }
    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool WrMutex::Unlock() const
{
    if ( pthread_mutex_unlock( m_pmutex ) == 0 )
    {
        return true;
    }
    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool WrMutex::TryLock() const
{
    if ( pthread_mutex_trylock( m_pmutex ) == 0U )
    {
        return true;
    }
    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool WrMutex::TryLock( unsigned long ms ) const
{
    time_t sec = (time_t)(ms/1000U);

    const timespec req = {sec, 0U};

    if ( pthread_mutex_timedlock(m_pmutex, &req) == 0U )
    {
        return true;
    }
    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cMutexLocker::cMutexLocker( WrMutex* mutex )
{
    m_mutex = mutex;
    if ( m_mutex )
    {
        m_mutex->Lock();
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cMutexLocker::~cMutexLocker()
{
    if ( m_mutex )
    {
        m_mutex->Unlock();
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
WrThreadEvent::WrThreadEvent() : m_isOwner( true )
{
    m_pcond = new pthread_cond_t;
    if ( pthread_cond_init( m_pcond, NULL ) == 0U )
    {
        //ok
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
WrThreadEvent::~WrThreadEvent()
{
    if ( m_isOwner )
    {
        switch ( pthread_cond_destroy( m_pcond ) )
        {
            case 0U:
                break;

            case EBUSY:
                pthread_cond_broadcast( m_pcond );
                pthread_cond_destroy( m_pcond );
                break;
        }

        delete m_pcond;
    }
}

/*****************************************************************************/
/**
  * @brief
  * @param copy Reference to original WrThreadEvent
 **
******************************************************************************/
WrThreadEvent::WrThreadEvent( const WrThreadEvent& copy )
{
    m_pcond = copy.m_pcond;
    m_isOwner = false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void WrThreadEvent::Notify()
{
    if ( pthread_cond_signal( m_pcond ) == 0U )
    {
        //ok
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void WrThreadEvent::NotifyAll()
{
    if ( pthread_cond_broadcast( m_pcond ) == 0U )
    {
        //ok
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void WrThreadEvent::Wait( WrMutex& mutex )
{
    if ( pthread_cond_wait( m_pcond, mutex.m_pmutex ) == 0U )
    {
        //ok
    }
}

/*****************************************************************************/
/**
  * @brief
  * @param mutex Reference to locked WrMutex
  * @param msec The time interval for which execution is to be suspended,
  *				and waits for a signal, in milliseconds
 **
******************************************************************************/
bool WrThreadEvent::Wait( WrMutex& mutex, long msec )
{
    timespec  timeout;
    timeval   timenow;

    gettimeofday( &timenow, 0U );

    long count_sec = msec / 1000U;
    long count_nsec = (msec % 1000U) * 1000000U;

    timeout.tv_sec = timenow.tv_sec + count_sec;
    timeout.tv_nsec = timenow.tv_usec * 1000U + count_nsec;

    if ( pthread_cond_timedwait( m_pcond, mutex.m_pmutex, &timeout ) == ETIMEDOUT )
    {
        return true;
    }
    else
    {
        return false;
    }

    return false;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
WrTimer::WrTimer( int timeout )
{
    m_timeout = timeout;

    Start();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
WrTimer::~WrTimer()
{
    TerminateProcessing();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool WrTimer::StartProcessing()
{
    return true;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool WrTimer::Processing()
{
    signalTimeout.Emit();

    Sleep( m_timeout );
    return true;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
bool WrTimer::EndProcessing()
{
    return true;
}
