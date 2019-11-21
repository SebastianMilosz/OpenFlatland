#include "performancelogger.hpp"

#include <utilities/MathUtilities.h>
#include <utilities/LoggerUtilities.h>
#include <utilities/SysUtilities.h>

#include <iostream>
#include <fstream>
#include <iomanip>

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
PerformanceLogger::PerformanceLogger() :
    m_applicationId(""),
    m_note("")
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
void PerformanceLogger::Initialize( const std::string& applicationId )
{
    m_applicationId = applicationId;
    LOGGER( LOG_INFO << "Performance Logger created for build: " << applicationId );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void PerformanceLogger::AddNote( const std::string& note )
{
    m_note = note;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void PerformanceLogger::SaveToFile( const std::string& filePath )
{
    std::ofstream performanceFile;
    performanceFile.open ( filePath, std::fstream::in | std::fstream::out | std::fstream::app );

    performanceFile << m_applicationId;

    for ( auto it = m_PerformanceMap.begin(); it != m_PerformanceMap.end(); it++ )
    {
        performanceFile << ", " << it->second.Name << " = " << std::fixed << std::setw(9)
        << std::setprecision(6) << it->second.Elapsed_ns/(10e8) << "s";
    }

    if( m_note.size() )
    {
        performanceFile << ", " << m_note;
    }
    performanceFile << '\n';

    performanceFile.close();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::string PerformanceLogger::PointToString( const unsigned int id )
{
    volatile double elapsed_ns( m_PerformanceMap[ id ].Elapsed_ns/(10e8) );

    std::ostringstream ss;

    ss << std::fixed << std::setw(9) << std::setprecision(6) << elapsed_ns;

    return  m_PerformanceMap[ id ].Name + std::string(" = ") + ss.str() + std::string("s");
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void PerformanceLogger::AddPerformancePoint( const unsigned int id, const std::string& name )
{
    m_PerformanceMap[ id ].Name = name;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void PerformanceLogger::PerformancePointEnter( const unsigned int id )
{
    m_timer.start();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void PerformanceLogger::PerformancePointLeave( const unsigned int id )
{
    m_PerformanceMap[ id ].Elapsed_ns = m_timer.get_elapsed_ns();
}
