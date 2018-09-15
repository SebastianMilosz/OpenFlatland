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
    m_applicationId("")
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
    m_applicationId = applicationId;
    LOGGER( LOG_INFO << "Performance Logger created for build: " << applicationId );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void PerformanceLogger::SaveToFile( std::string filePath )
{
    std::ofstream performanceFile;
    performanceFile.open ( filePath, std::fstream::in | std::fstream::out | std::fstream::app );

    performanceFile << m_applicationId;

    std::map<unsigned int , PerformanceData>::iterator it;
    for ( it = m_PerformanceMap.begin(); it != m_PerformanceMap.end(); it++ )
    {
        performanceFile << ", " << it->second.Name << " = " << std::fixed << std::setw(9)
        << std::setprecision(6) << it->second.Elapsed_ns/(10e8) << "s";
    }

    performanceFile << '\n';

    performanceFile.close();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
std::string PerformanceLogger::PointToString( unsigned int id )
{
    double elapsed_ns = m_PerformanceMap[ id ].Elapsed_ns/(10e8);
    return  m_PerformanceMap[ id ].Name + std::string(" = ") + utilities::math::DoubleToStr( elapsed_ns ) + std::string("s");
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void PerformanceLogger::AddPerformancePoint( unsigned int id, std::string name )
{
    m_PerformanceMap[ id ].Name = name;
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void PerformanceLogger::PerformancePointEnter( unsigned int id )
{
    m_timer.start();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void PerformanceLogger::PerformancePointLeave( unsigned int id )
{
    m_PerformanceMap[ id ].Elapsed_ns = m_timer.get_elapsed_ns();
}
