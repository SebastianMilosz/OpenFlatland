#ifndef WORLD_H
#define WORLD_H

#include <Box2D/Box2D.h>
#include <SFML/Graphics.hpp>
#include <serializable.hpp>

#include "entity.hpp"
#include "constelement.hpp"

class World : public codeframe::cSerializable
{
    public:
        std::string Role()            const { return "Object"; }
        std::string Class()           const { return "World";  }
        std::string BuildType()       const { return "Static"; }
        std::string ConstructPatern() const { return "";       }

    public:
                 World( std::string name, cSerializableInterface* parent );
        virtual ~World();

        void AddShell( std::shared_ptr<Entity>       entity );
        void AddConst( std::shared_ptr<ConstElement> constElement );

        bool PhysisStep();
        bool Draw( sf::RenderWindow& window );

        void MouseDown( float x, float y );
        void MouseUp( float x, float y );
        void MouseMove( float x, float y );

    protected:

    private:
        b2Body* getBodyAtMouse( float x, float y );

        b2Body*         m_GroundBody;
        b2MouseJoint*   m_MouseJoint;
        b2MouseJointDef m_JointDef;
        b2Vec2          m_Gravity;
        b2World         m_World;

        bool            m_entitySelMode;
};

#endif // WORLD_H
