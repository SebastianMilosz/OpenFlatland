#ifndef ENTITYSHELL_HPP
#define ENTITYSHELL_HPP

#include "physics_body.hpp"
#include "drawable_entity_motion.hpp"
#include "drawable_entity_vision.hpp"

class EntityShell : public PhysicsBody
{
        CODEFRAME_META_CLASS_NAME( "EntityShell" );
        CODEFRAME_META_BUILD_TYPE( codeframe::DYNAMIC );
        CODEFRAME_META_CONSTRUCT_PATERN( "X,Y" );

    public:
                 EntityShell( const std::string& name, int x, int y );
                 EntityShell( const EntityShell& other );
        virtual ~EntityShell();

        codeframe::Property<int        > X;
        codeframe::Property<int        > Y;
        codeframe::Property<float      > Rotation;
        codeframe::Property<std::string> Name;
        codeframe::Property<float      > Density;
        codeframe::Property<float      > Friction;

        EntityShell& operator=(const EntityShell& other);

        void Draw( sf::RenderWindow& window ) override;
        void Synchronize( b2Body* body ) override;

        const int& GetX();
        float32 GetPhysicalX();
        void SetX(int val);
        const int& GetY();
        float32 GetPhysicalY();
        void SetY(int val);

        const float32& GetRotation();

        void SetRotation( float rotation );

        const b2Vec2& GetPhysicalPoint();

        EntityVision& Vision() { return m_vision; }

    private:
        b2Vec2               m_zeroVector;
        sf::CircleShape      m_triangle;
        DrawableEntityVision m_vision;
        DrawableEntityMotion m_motion;

        int     m_curX;
        int     m_curY;
        float32 m_curR;

        void slotSelectionChanged( smart_ptr<ObjectNode> );
};

#endif // ENTITYSHELL_HPP
