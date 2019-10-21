#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include <iostream>
#include <conio.h>

int main(int argc, char *argv[])
{
    int result = Catch::Session().run( argc, argv );

    do
    {
        std::cout << '\n' << "Press a key to continue...";
    } while ( std::cin.get() != '\n' );

    return ( result < 0xff ? result : 0xff );
}
