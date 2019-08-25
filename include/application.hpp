#ifndef APPLICATION_HPP_INCLUDED
#define APPLICATION_HPP_INCLUDED

#include <serializable.hpp>

#include "world.hpp"
#include "entityfactory.hpp"
#include "constelementsfactory.hpp"
#include "fontfactory.hpp"
#include "guiwidgetslayer.hpp"
#include "entity.hpp"

class Application : public codeframe::cSerializable
{
        CODEFRAME_META_CLASS_NAME( "Application" );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

    public:
                 Application( std::string name, sf::RenderWindow& window );
        virtual ~Application();

        void ProcesseEvents( sf::Event& event );
        void ProcesseLogic( void );

    private:
            class AplicationDataStorage : public utilities::data::DataStorage
            {
                public:
                    AplicationDataStorage( const std::string& filePath ) :
                        m_filePath( filePath )
                    {

                    }

                   virtual void Add( const std::string& key, const std::string& value )
                   {

                   }

                   virtual void Get( const std::string& key, std::string& value )
                   {

                   }
                private:
                    std::string m_filePath;
            };

            void ZoomViewAt( sf::Vector2i pixel, sf::RenderWindow& window, float zoom );

            const float         m_zoomAmount;

            std::string         m_apiDir;
            std::string         m_cfgFilePath;
            std::string         m_perFilePath;
            std::string         m_logFilePath;
            std::string         m_guiFilePath;

            AplicationDataStorage   m_AplicationDataStorage;

            sf::RenderWindow&       m_Window;
            GUIWidgetsLayer         m_Widgets;
            World                   m_World;
            EntityFactory           m_EntityFactory;
            ConstElementsFactory    m_ConstElementsFactory;
            FontFactory             m_FontFactory;

            // Temporary
            int lineCreateState;
            sf::Vector2f startPoint;
            sf::Vector2f endPoint;
};

#endif // APPLICATION_HPP_INCLUDED
