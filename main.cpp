#include <iostream>

#include <utilities/LoggerUtilities.h>
#include <utilities/FileUtilities.h>
#include <serializable.hpp>
#include <SFML/Graphics.hpp>
#include <cpgf/gcallbacklist.h>
#include <Box2D/Box2D.h>

#include "application.hpp"
#include "logwidget.hpp"

int main()
{
    sf::RenderWindow window(sf::VideoMode(800, 600, 32), "" );
    Application      application( "Application", window );

    while ( window.isOpen() )
    {
        sf::Event event;
        while ( window.pollEvent( event ) )
        {
            application.ProcesseEvents( event );
        }

        application.ProcesseLogic();
    }

    return 0;
}
