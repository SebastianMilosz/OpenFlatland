#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include <utilities/LoggerUtilities.h>

#include <iostream>
#include <conio.h>

int main(int argc, char *argv[])
{
    const std::string apiDir( utilities::file::GetExecutablePath() );
    const std::string logFilePath( apiDir + std::string("\\test_log.txt") );

    LOGGERINS().LogPath = logFilePath;

    LOGGER( LOG_INFO << "Begin Catch Test Session" );

    int result = Catch::Session().run( argc, argv );

    do
    {
        std::cout << '\n' << "Press a key to continue...";
    } while ( std::cin.get() != '\n' );

    return ( result < 0xff ? result : 0xff );
}
