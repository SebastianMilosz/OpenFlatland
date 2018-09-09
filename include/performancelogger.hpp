#ifndef PERFORMANCELOGGER_HPP_INCLUDED
#define PERFORMANCELOGGER_HPP_INCLUDED

#include <string>
#include <map>

#define PERFORMANCE_INITIALIZE(p1) PerformanceLogger::GetInstance().Initialize(p1)
#define PERFORMANCE_ADD(p1,p2) PerformanceLogger::GetInstance().AddPerformancePoint(p1,p2)
#define PERFORMANCE_ENTER(p1) PerformanceLogger::GetInstance().PerformancePointEnter(p1)
#define PERFORMANCE_LEAVE(p1) PerformanceLogger::GetInstance().PerformancePointLeave(p1)
#define PERFORMANCE_SAVE(p1) PerformanceLogger::GetInstance().SaveToFile(p1)

class PerformanceLogger
{
    public:
        struct PerformanceData
        {
            std::string Name;
            double      StartTime;
            double      EndTime;
        };

    public:
        ~PerformanceLogger();

        static PerformanceLogger& GetInstance();

        void Initialize( std::string applicationId );
        void SaveToFile( std::string filePath );

        void AddPerformancePoint( unsigned int id, std::string name );

        void PerformancePointEnter( unsigned int id );
        void PerformancePointLeave( unsigned int id );

    private:
        PerformanceLogger();

        std::map<unsigned int , PerformanceData> m_PerformanceMap;
};

#endif // PERFORMANCELOGGER_HPP_INCLUDED
