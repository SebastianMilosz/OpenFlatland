#ifndef ENTITYSHELL_H
#define ENTITYSHELL_H

#include <Box2D/Box2D.h>

class EntityShell
{
    public:
        EntityShell();
        virtual ~EntityShell();
        EntityShell(const EntityShell& other);
        EntityShell& operator=(const EntityShell& other);

        unsigned int Getx() { return m_Body->GetPosition().x; }
        void Setx(unsigned int val) { m_x = val; }
        unsigned int Gety() { return m_y; }
        void Sety(unsigned int val) { m_y = val; }
        unsigned int Getz() { return m_z; }
        void Setz(unsigned int val) { m_z = val; }

    protected:

    private:
        b2Body* m_Body;
};

#endif // ENTITYSHELL_H
