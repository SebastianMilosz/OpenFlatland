#include "utilities/PerformanceUtilities.h"
#include "utilities/SysUtilities.h"
#include <sys/time.h>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
cPerformanceCounter::cPerformanceCounter( std::string name ) :
    m_name(name)
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cPerformanceCounter::RegisterCounter( int id, std::string cntname )
{
    m_counterName[id] = cntname;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cPerformanceCounter::UpdateCounterStart( int id )
{
    double stime           = utilities::system::GetTime();
    m_counterTempValue[id] = stime;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void cPerformanceCounter::UpdateCounterEnd( int id )
{
    double etime       = utilities::system::GetTime();
    m_counterValue[id] = (etime - m_counterValue[id]);
}
