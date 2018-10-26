#ifndef ENTITYSHELL_HPP
#define ENTITYSHELL_HPP

#include "physicsbody.hpp"
#include "entityvision.hpp"

class EntityShell : public PhysicsBody
{
        CODEFRAME_META_CLASS_NAME( "EntityShell" );
        CODEFRAME_META_BUILD_TYPE( codeframe::DYNAMIC );
        CODEFRAME_META_CONSTRUCT_PATERN( "X,Y" );

    public:
                 EntityShell( std::string name, int x, int y );
                 EntityShell( const EntityShell& other );
        virtual ~EntityShell();

        codeframe::Property<int,          EntityShell> X;
        codeframe::Property<int,          EntityShell> Y;
        codeframe::Property<float,        EntityShell> Rotation;
        codeframe::Property<bool,         EntityShell> CastRays;
        codeframe::Property<unsigned int, EntityShell> RaysCnt;
        codeframe::Property<unsigned int, EntityShell> RaysSize;
        codeframe::Property<std::string,  EntityShell> Name;
        codeframe::Property<float,        EntityShell> Density;
        codeframe::Property<float,        EntityShell> Friction;

        EntityShell& operator=(const EntityShell& other);

        virtual void Draw( sf::RenderWindow& window, b2Body* body );

        const int& GetX();
        float32 GetPhysicalX();
        void SetX(int val);
        const int& GetY();
        float32 GetPhysicalY();
        void SetY(int val);

        const float32& GetRotation();

        const b2Vec2& GetPhysicalPoint();

        EntityVision& Vision() { return m_vision; }

    private:
        b2Vec2          m_zeroVector;
        sf::CircleShape m_circle;
        sf::CircleShape m_triangle;
        EntityVision    m_vision;

        int     m_curX;
        int     m_curY;
        float32 m_curR;

        void slotSelectionChanged( smart_ptr<cSerializableInterface> );
};

#endif // ENTITYSHELL_HPP
