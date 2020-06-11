#ifndef SERIALIZABLE_CHILD_LIST_H
#define SERIALIZABLE_CHILD_LIST_H

#include <vector>
#include <MathUtilities.h>
#include <ThreadUtilities.h>
#include <smartpointer.h>

namespace codeframe
{
    class ObjectNode;
    class ObjectSelection;

    class cObjectList
    {
        friend class iterator;

        public:
            class iterator : public std::iterator< std::input_iterator_tag, smart_ptr<ObjectNode> >
            {
                friend class cObjectList;

                private:
                    int          m_childCnt;
                    cObjectList* m_childList;
                    smart_ptr<ObjectNode>  m_serializable;

                public:
                    iterator(const iterator& n) : m_childCnt(n.m_childCnt), m_childList(n.m_childList), m_serializable(n.m_serializable) {}

                    // Operator wskaznikowy wyodrebnienia wskazywanej wartosci
                    smart_ptr<ObjectNode> operator *()
                    {
                        m_serializable = m_childList->m_childVector.at( m_childCnt );
                        return m_serializable;
                    }

                    operator int() const
                    {
                        return m_childCnt;
                    }

                    // Operator inkrementacji (przejscia na kolejne pole)
                    iterator& operator ++(){ if(m_childCnt < m_childList->size()) ++m_childCnt; return *this; }

                    // Operatory porownania
                    bool operator< (const iterator& n) { return  (n.m_childCnt <  m_childCnt); }
                    bool operator> (const iterator& n) { return  (n.m_childCnt >  m_childCnt); }
                    bool operator<=(const iterator& n) { return !(n.m_childCnt >  m_childCnt); }
                    bool operator>=(const iterator& n) { return !(n.m_childCnt <  m_childCnt); }
                    bool operator==(const iterator& n) { return  (n.m_childCnt == m_childCnt); }
                    bool operator!=(const iterator& n) { return !(n.m_childCnt == m_childCnt); }

                private:
                    iterator(cObjectList* b, int n) : m_childCnt(n), m_childList(b) {}
            };

        public:
            cObjectList();
            void Register  ( smart_ptr<ObjectNode> child );
            void UnRegister( smart_ptr<ObjectNode> child );
            void PulseChanged( bool fullTree = false );
            void CommitChanges();
            void Enable( bool val );

            smart_ptr<ObjectSelection> GetObjectByName( const std::string& name );
            smart_ptr<ObjectSelection> GetObjectById  ( const uint32_t id   );

            // Iterator
            iterator begin() throw()      { return iterator(this, 0);                    }
            iterator end()   throw()      { return iterator(this, m_childVector.size()); }
            std::string      name()        const { return std::string("ChildList");             }
            int              size()        const { return m_childVector.size();                 }
            std::string      sizeString()  const { return utilities::math::IntToStr(size());    }

        private:
            std::vector< smart_ptr<ObjectNode> > m_childVector;

            WrMutex m_Mutex;
    };

}

#endif // SERIALIZABLE_CHILD_LIST_H
