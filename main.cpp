#include <box2d/box2d.h>
#include <SFML/Graphics.hpp>

#include <utilities/LoggerUtilities.h>
#include <utilities/FileUtilities.h>
#include <serializable_object.hpp>

#include <iostream>

#include "application.hpp"
#include "console_widget.hpp"

int main()
{
    sf::RenderWindow window(sf::VideoMode(sf::Vector2u(800, 600)), "" );

    Application      application("Application", window);

    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // Enable Docking
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;       // Enable Multi-Viewport / Platform Windows

    while (window.isOpen())
    {
        application.ProcesseLogic();

        while (const std::optional event = window.pollEvent())
        {
            application.ProcesseEvents(event);
        }
    }

    return 0;
}
