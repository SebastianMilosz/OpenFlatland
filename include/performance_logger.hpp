#ifndef PERFORMANCELOGGER_HPP_INCLUDED
#define PERFORMANCELOGGER_HPP_INCLUDED

#include <string>
#include <map>

#include <plf_nanotimer.h>

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
            PerformanceData() :
                Name(),
                Elapsed_ns( 0 )
            {

            }

            PerformanceData( const std::string& name ) :
                Name( name ),
                Elapsed_ns( 0 )
            {

            }

            std::string Name;
            double Elapsed_ns;
        };

    public:
        ~PerformanceLogger() = default;

        static PerformanceLogger& GetInstance();

        void Initialize( const std::string& applicationId );
        void AddNote( const std::string& note );
        void SaveToFile( const std::string& filePath );

        std::string PointToString( const unsigned int id );

        void AddPerformancePoint( const unsigned int id, const std::string& name );

        void PerformancePointEnter( const unsigned int id );
        void PerformancePointLeave( const unsigned int id );

    private:
        PerformanceLogger();

        plf::nanotimer m_timer;
        std::string m_applicationId;
        std::string m_note;
        std::map<unsigned int , PerformanceData> m_PerformanceMap;
};

#endif // PERFORMANCELOGGER_HPP_INCLUDED
