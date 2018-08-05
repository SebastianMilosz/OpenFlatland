#ifndef CSERIALIZABLECONTAINER_H
#define CSERIALIZABLECONTAINER_H

#include <serializable.h>
#include <MathUtilities.h>
#include <exception>
#include <stdexcept>
#include <vector>
#include <cstdbool>
#include <smartpointer.h>

#define MAXID 100

namespace codeframe
{

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    class cIgnoreList
    {
    public:
        cIgnoreList() { }

        struct sIgnoreEntry
        {
            sIgnoreEntry() : Name(""), ClassName(""), BuildType(""), Ignore(false) {}
            sIgnoreEntry(std::string name, std::string className = "", std::string buildType = "", bool ignore = true) :
                Name(name), ClassName(className), BuildType(buildType), Ignore(ignore) {}

            std::string Name;
            std::string ClassName;
            std::string BuildType;
            bool Ignore;
        };

        void AddToList( cSerializableInterface* serObj, bool ignore = true )
        {
            if( serObj )
            {
                m_vectorIgnoreEntry.push_back( sIgnoreEntry( serObj->ObjectName(), serObj->Class(), serObj->BuildType(), ignore ) );
            }
        }

        void AddToList( std::string name = "", std::string className = "", std::string buildType = "", bool ignore = true )
        {
            m_vectorIgnoreEntry.push_back( sIgnoreEntry( name, className, buildType, ignore ) );
        }

        bool IsIgnored( cSerializableInterface* serObj )
        {
            if( m_vectorIgnoreEntry.empty() == false && serObj )
            {
                for( std::vector<sIgnoreEntry>::iterator it = m_vectorIgnoreEntry.begin(); it != m_vectorIgnoreEntry.end(); ++it )
                {
                    sIgnoreEntry entry = *it;

                    /* std::cout << *it; ... */
                    if( entry.Name == serObj->ObjectName() && entry.BuildType == serObj->BuildType() && entry.Ignore )
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        void Dispose()
        {
            m_vectorIgnoreEntry.clear();
        }

    private:
        std::vector<sIgnoreEntry> m_vectorIgnoreEntry;

    };

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    class cSerializableContainer : public cSerializable
    {
        private:
            unsigned int m_select;
            unsigned int m_size;

        public:
            std::string Role()      const { return "Container"; }
            std::string Class()     const { return "cSerializableContainer"; }
            std::string BuildType() const { return "Static"; }

        public:
            cSerializableContainer( std::string name, cSerializableInterface* parentObject ) :  cSerializable( name, parentObject ), m_select(0), m_size( 0 )
            {
            }

            virtual ~cSerializableContainer()
            {
                Dispose();
            }

            int Count() const
            {
                return m_size;
            }

            virtual smart_ptr<cSerializableInterface> Create(
                                                             const std::string className,
                                                             const std::string objName,
                                                             const std::vector<codeframe::VariantValue>& params = std::vector<codeframe::VariantValue>()
                                                             ) = 0;

            /*****************************************************************************/
            /**
              * @brief
             **
            ******************************************************************************/
            virtual void CreateRange( std::string className, std::string objName, int range )
            {
                for(int i = 0; i < range; i++)
                {
                    std::string objNameNum = objName + utilities::math::IntToStr( i );

                    if( smart_ptr_isValid( Create( className, objNameNum ) ) == false )
                    {
                        throw std::runtime_error( "cSerializableContainer::Create return NULL" );
                    }
                }
            }

            /*****************************************************************************/
            /**
              * @brief
             **
            ******************************************************************************/
            bool IsName( std::string& name )
            {
                for(typename std::vector< smart_ptr<cSerializableInterface> >::iterator it = m_containerVector.begin(); it != m_containerVector.end(); ++it)
                {
                    smart_ptr<cSerializableInterface> sptr = *it;

                    if( smart_ptr_isValid( sptr ) == true )
                    {
                        std::string inContainerName = sptr->ObjectName();

                        if( name == inContainerName )
                        {
                            return true;
                        }
                    }
                }

                return false;
            }

            /*****************************************************************************/
            /**
              * @brief
             **
            ******************************************************************************/
            std::string CreateUniqueName( std::string nameBase )
            {
                std::string uniqueName  = nameBase;

                for( int curIter = 0; curIter < MAXID; curIter++ )
                {
                    std::string name = nameBase + utilities::math::IntToStr( curIter );

                    if( IsName( name ) == false )
                    {
                        uniqueName = name;
                        break;
                    }
                }

                return uniqueName;
            }

            /*****************************************************************************/
            /**
              * @brief
             **
            ******************************************************************************/
            virtual bool Dispose( unsigned int objId )
            {
                if( m_containerVector.size() <= objId ) return false;

                smart_ptr<cSerializableInterface> obj = m_containerVector[ objId ];

                if( obj )
                {
                    m_containerVector[ objId ] = smart_ptr<cSerializableInterface>(NULL);
                    if( m_size ) m_size--;
                    return true;
                }

                return false;
            }

            /*****************************************************************************/
            /**
              * @brief
             **
            ******************************************************************************/
            virtual bool Dispose( std::string objName )
            {
                objName = "";
                return false;
            }

            /*****************************************************************************/
            /**
              * @brief
             **
            ******************************************************************************/
            virtual bool DisposeByBuildType( std::string serType, cIgnoreList ignore = cIgnoreList() )
            {
                for(typename std::vector< smart_ptr<cSerializableInterface> >::iterator it = m_containerVector.begin(); it != m_containerVector.end();)
                {
                    smart_ptr<cSerializableInterface> sptr = *it;

                    if( smart_ptr_isValid( sptr ) && sptr->BuildType() == serType && ignore.IsIgnored( smart_ptr_getRaw( sptr ) ) == false )
                    {
                        *it = smart_ptr<cSerializableInterface>(NULL);
                        if( m_size ) m_size--;
                        signalSelected.Emit( m_select );
                    }
                    else
                    {
                        it++;
                    }
                }

                return true;
            }

            /*****************************************************************************/
            /**
              * @brief
             **
            ******************************************************************************/
            template<typename T>
            bool Dispose( smart_ptr<T> obj )
            {
                for(typename std::vector< smart_ptr<T> >::iterator it = m_containerVector.begin(); it != m_containerVector.end(); ++it)
                {
                    smart_ptr<T> sptr = *it;

                    if( smart_ptr_isValid( sptr ) && smart_ptr_isValid( obj ) )
                    {
                        if( sptr->ObjectName() == obj->ObjectName() )
                        {
                            *it = smart_ptr<T>();
                            if( m_size ) m_size--;
                            signalSelected.Emit( m_select );
                            return true;
                        }
                    }
                }

                return false;
            }

            /*****************************************************************************/
            /**
              * @brief
             **
            ******************************************************************************/
            virtual bool Dispose()
            {
                if( m_containerVector.size() == 0 ) return true;    // Pusty kontener zwracamy prawde bo nie ma nic do usuwania

                for(typename std::vector< smart_ptr<cSerializableInterface> >::iterator it = m_containerVector.begin(); it != m_containerVector.end(); ++it)
                {
                    smart_ptr<cSerializableInterface> obj = *it;

                    // Usuwamy tylko jesli nikt inny nie korzysta z obiektu
                    if( smart_ptr_getCount( obj ) <= 2 )
                    {
                        obj = smart_ptr<cSerializableInterface>(NULL);
                    }
                    else // Nie mozna usunac obiektu
                    {
                        return false;
                    }
                }

                m_containerVector.clear();
                m_size = 0;

                return true;
            }

            /*****************************************************************************/
            /**
              * @brief
             **
            ******************************************************************************/
            bool IsInRange( unsigned int cnt ) const
            {
                return (bool)( cnt < m_containerVector.size() );
            }

            /*****************************************************************************/
            /**
              * @brief
             **
            ******************************************************************************/
            template<typename T>
            smart_ptr<T> operator[]( int i )
            {
                return Get<T>( i );
            }

            /*****************************************************************************/
            /**
              * @brief
             **
            ******************************************************************************/
            bool Select( int pos )
            {
                if( IsInRange( pos ) )
                {
                    m_select = pos;
                    signalSelected.Emit( m_select );
                    return true;
                }
                return false;
            }

            /*****************************************************************************/
            /**
              * @brief
             **
            ******************************************************************************/
            bool IsSelected()
            {
                return IsInRange( m_select );
            }

            /*****************************************************************************/
            /**
              * @brief
             **
            ******************************************************************************/
            template<typename T>
            smart_ptr<T> GetSelected()
            {
                if( IsInRange( m_select ) )
                {
                    return Get<T>( m_select );
                }

                throw std::out_of_range( "cSerializableContainer::GetSelected(" + utilities::math::IntToStr(m_select) + "): Out of range" );

                return smart_ptr<T>();
            }

            /*****************************************************************************/
            /**
              * @brief
             **
            ******************************************************************************/
            int GetSelectedPosition() const
            {
                return m_select;
            }

            /*****************************************************************************/
            /**
              * @brief
             **
            ******************************************************************************/
            template<typename T>
            smart_ptr<T> Get( int id )
            {
                if( IsInRange( id ) )
                {
                    smart_ptr<T> obj = m_containerVector.at( id );

                    if( obj.IsValid() )
                    {
                        return obj;
                    }
                }

                throw std::out_of_range( "cSerializableContainer::Get(" + utilities::math::IntToStr(id) + "): Out of range" );

                return NULL;
            }

            /*****************************************************************************/
            /**
              * @brief
             **
            ******************************************************************************/
            int Add( smart_ptr<cSerializableInterface> classType, int pos = -1 )
            {
                return InsertObject( classType, pos );
            }

            signal1<int> signalSelected;

        protected:
            /*****************************************************************************/
            /**
              * @brief
             **
            ******************************************************************************/
            virtual int InsertObject( smart_ptr<cSerializableInterface> classType, int pos = -1 )
            {
                // pos == -1 oznacza pierwszy lepszy
                bool found  = false;
                int  retPos = -1;

                if( pos < static_cast<int>(m_containerVector.size()) )
                {
                    // Szukamy bezposrednio
                    if( pos >= 0 )
                    {
                        smart_ptr<cSerializableInterface> tmp = m_containerVector[ pos ];

                        if( smart_ptr_isValid( tmp ) )
                        {
                            m_containerVector[ pos ] = classType;
                            found = true;
                        }
                    }

                    // Szukamy wolnego miejsca
                    if( found == false )
                    {
                        // Po calym wektorze szukamy pustych miejsc
                        for(typename std::vector< smart_ptr<cSerializableInterface> >::iterator it = m_containerVector.begin(); it != m_containerVector.end(); ++it)
                        {
                            smart_ptr<cSerializableInterface> obj = *it;

                            if( smart_ptr_isValid( obj ) == false )
                            {
                                // Znalezlismy wiec zapisujemy
                                *it = classType;
                                found = true;
                                retPos = std::distance( m_containerVector.begin(), it );
                                break;
                            }
                        }
                    }
                }

                // poza zakresem dodajemy do wektora nowa pozycje
                if( found == false )
                {
                    m_containerVector.push_back( classType );
                    retPos = m_containerVector.size() - 1;
                }

                // If there is no parent we become one
                if( NULL == classType->Parent() )
                {
                    cSerializableInterface* serPar = static_cast<cSerializableInterface*>( this );
                    classType->ParentBound( serPar );
                }

                m_select = retPos;
                m_size++;

                classType->SetId( m_select );

                // Emitujemy sygnal zmiany selekcji
                signalSelected.Emit( m_select );
                return retPos;
            }

        private:
            std::vector< smart_ptr<cSerializableInterface> > m_containerVector;
    };

}

#endif // CSERIALIZABLECONTAINER_H
