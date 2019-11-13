#ifndef SERIALIZABLECHILDLIST_H
#define SERIALIZABLECHILDLIST_H

#include <vector>
#include <MathUtilities.h>
#include <ThreadUtilities.h>

namespace codeframe
{
    class ObjectNode;

    class cSerializableChildList
    {
        friend class iterator;

        private:
            int                                  m_childCnt;
            std::vector<ObjectNode*> m_childVector;

            ///
            WrMutex m_Mutex;
        public:
            class iterator : public std::iterator<std::input_iterator_tag, ObjectNode*>
            {
                friend class cSerializableChildList;

                private:
                    int                     m_childCnt;
                    cSerializableChildList* m_childList;
                    ObjectNode* m_serializable;

                public:
                    iterator(const iterator& n) : m_childCnt(n.m_childCnt), m_childList(n.m_childList), m_serializable(n.m_serializable) {}

                    // Operator wskaznikowy wyodrebnienia wskazywanej wartosci
                    ObjectNode* operator *()
                    {
                        m_serializable = m_childList->m_childVector.at( m_childCnt );
                        return m_serializable;
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
                    iterator(cSerializableChildList* b, int n) : m_childCnt(n), m_childList(b) {}
            };

        public:
            cSerializableChildList();
            void Register  ( ObjectNode* child );
            void UnRegister( ObjectNode* child );
            void PulseChanged( bool fullTree = false );
            void CommitChanges();
            void Enable( bool val );

            // Iterator
            iterator begin() throw()      { return iterator(this, 0);                    }
            iterator end()   throw()      { return iterator(this, m_childVector.size()); }
            std::string      name()        const { return std::string("ChildList");             }
            int              size()        const { return m_childVector.size();                 }
            std::string      sizeString()  const { return utilities::math::IntToStr(size());    }

    };

}

#endif // SERIALIZABLECHILDLIST_H
