#include "performancelogger.hpp"

#include <utilities/MathUtilities.h>
#include <utilities/LoggerUtilities.h>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
PerformanceLogger::PerformanceLogger()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
PerformanceLogger::~PerformanceLogger()
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
PerformanceLogger& PerformanceLogger::GetInstance()
{
    static PerformanceLogger Instance;

    return Instance;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void PerformanceLogger::Initialize( std::string applicationId )
{
    LOGGER( LOG_INFO << "Performance Logger created for build: " << applicationId );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void PerformanceLogger::SaveToFile( std::string filePath )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void PerformanceLogger::AddPerformancePoint( unsigned int id, std::string name )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void PerformanceLogger::PerformancePointEnter( unsigned int id )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void PerformanceLogger::PerformancePointLeave( unsigned int id )
{

}
