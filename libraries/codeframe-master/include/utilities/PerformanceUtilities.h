#ifndef PERFORMANCECOUNTER_H
#define PERFORMANCECOUNTER_H

#define PERFORMANCECOUNTER cPerformanceCounter pc;
#define PCNT_START(x) pc.UpdateCounterStart(x);
#define PCNT_END(x) pc.UpdateCounterEnd(x);

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

class cPerformanceCounter
{
private:
    std::string                 m_name;
    std::vector<std::string>    m_counterName;
    std::vector<double>         m_counterValue;
    std::vector<double>         m_counterTempValue;
public:
    cPerformanceCounter( std::string name );

    void RegisterCounter   ( int id, std::string cntname );
    void UpdateCounterStart( int id );
    void UpdateCounterEnd  ( int id );
};

#endif // PERFORMANCECOUNTER_H
