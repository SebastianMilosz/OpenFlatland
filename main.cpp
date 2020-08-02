#include <Box2D/Box2D.h>
#include <SFML/Graphics.hpp>

#include <utilities/LoggerUtilities.h>
#include <utilities/FileUtilities.h>
#include <serializable_object.hpp>

#include <iostream>

#include "application.hpp"
#include "console_widget.hpp"

int main()
{
    sf::RenderWindow window(sf::VideoMode(800, 600, 32), "" );
    Application      application( "Application", window );

    while ( window.isOpen() )
    {
        application.ProcesseLogic();

        sf::Event event;
        while ( window.pollEvent( event ) )
        {
            application.ProcesseEvents( event );
        }
    }

    return 0;
}
