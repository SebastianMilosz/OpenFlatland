#ifndef MIOCANPROTOCOL_H
#define MIOCANPROTOCOL_H

#include <serializableproperty.h>

namespace codeframe
{

#define REG_COMMAND         (0xFB)
#define CAN_BASE_POOL_TYPE  (0x80)

PACK_STRUCT_BEGIN
struct sMioObjProtFrame
{
    PACK_STRUCT_FIELD( uint8_t Adr  );
    PACK_STRUCT_FIELD( uint8_t Typ  );
    PACK_STRUCT_FIELD( uint8_t Radr );
    PACK_STRUCT_FIELD( uint8_t Dat0 );
    PACK_STRUCT_FIELD( uint8_t Dat1 );
    PACK_STRUCT_FIELD( uint8_t Dat2 );
    PACK_STRUCT_FIELD( uint8_t Dat3 );
    PACK_STRUCT_FIELD( uint8_t Dat4 );
} PACK_STRUCT_STRUCT;
PACK_STRUCT_END

// Adresy bazowe (niezmienne) modulow (tylko 4 pierwsze bity pozostale to zmienna czesc adresu)
#define ADR_MASTER                  (0xFE)   ///< Adres Mastera
#define ADR_IOMOD                   (0x01)   ///< Adres mod. Wejsc Wyjsc

// Typy RAMKI Pobrania Maksymalna Wartosc 0x0F (TYLKO 4 bity)
#define PACKET_EMPTY                (0x00)   ///< Brak typu mozna stosowac do wysylania pinga
#define PACKET_RDREG                (0x01)   ///< Ramka to rzadanie odczytu
#define PACKET_WRREG                (0x02)   ///< Ramka to rzadania zapisu

// Typy RAMKI Odpowiedzi
#define PACKET_RETEMPTY             (0x03)
#define PACKET_RETRDREG             (0x04)
#define PACKET_RETWRREG             (0x05)

// Pola ramki Protokolu
#define FRAME_ADR                   (0x00)    ///< Adres Nadawcy 0x00 - MASTER
#define FRAME_TYPE                  (0x01)    ///< [0-2] Rodaj ramki: Zapis/Odczyt/OdpZapis/OdpOdczyt [3-7] Maska Pola Danych
#define FRAME_RADR                  (0x02)    ///< Adres Odczytywanego rejestru
#define FRAME_DAT0                  (0x03)    ///< Pole Danych 0
#define FRAME_DAT1                  (0x04)    ///< Pole Danych 1
#define FRAME_DAT2                  (0x05)    ///< Pole Danych 2
#define FRAME_DAT3                  (0x06)    ///< Pole Danych 3
#define FRAME_DAT4                  (0x07)    ///< Pole Danych 4

// Rejestry Wspolne dla wszystkich modulow [ Kazdy rejestr to 5 Bajtow ]
#define REG_DerviceType             (0x00)    ///< [0] Typ [1-2] HardVer [3-4] SoftVer
#define REG_ProductId               (0x01)    ///< [0-3] Numer kolejny produktu [4] Nr klienta
#define REG_DerviceText             (0x02)    ///< Tekst Nazwa Zarezerwowana 'PSIIO'
#define REG_DerviceStat0            (0x03)    ///< Status [0] Stan Podstawowy (0x00 dziala dowolny bit ustawiony oznacza jakis blad)

// Rejestry Dedykowane Modul Wejsc Wyjsc

#define REG_DerviceStat1            (0x04)    ///< Status konkretnego modulu

#define REG_MODT_1_1                (0x05)    ///< Pierwsze 4 Bajty
#define REG_MODT_1_2                (0x06)    ///< Drugie 4 Bajty
#define REG_MODT_2_1                (0x07)
#define REG_MODT_2_2                (0x08)
#define REG_MODT_3_1                (0x09)
#define REG_MODT_3_2                (0x0A)
#define REG_MODT_4_1                (0x0B)
#define REG_MODT_4_2                (0x0C)

// Rejestry Wartosci Modulu (interpretowac w kontekscie typu)
#define REG_MODV_1_1                (0x0D)
#define REG_MODV_1_2                (0x0E)
#define REG_MODV_2_1                (0x0F)
#define REG_MODV_2_2                (0x10)
#define REG_MODV_3_1                (0x11)
#define REG_MODV_3_2                (0x12)
#define REG_MODV_4_1                (0x13)
#define REG_MODV_4_2                (0x14)

#define REG_MODV                    (0x15)    // Zapis odczyt wartosci z konkretneg gniazda REG_MODV(slot, socket, value)

#define REG_SYNINPUTINFO_A          (0x16)	    ///< Odczyt informacji z wejsc synchronicznych Gr_A
#define REG_SYNINPUTINFO_B          (0x17)	    ///< Odczyt informacji z wejsc synchronicznych Gr_B
#define REG_SOCKET_CONFIG_FLAG      (0x18)      ///< Rejestr flagi konfiguracyjnej okreslonego socketu
#define REG_SOCKET_PDICTIONARY 	    (0x19)		///< Rejestr pobierania slownika powiazania slot/socket -> globalsocket

#define PACK_STRUCT_BEGIN
#define PACK_STRUCT_STRUCT __attribute__ ((__packed__))
#define PACK_STRUCT_END
#define PACK_STRUCT_FIELD(x) x

#define REG_CTRLMATRIX0			(0xE5)  			///< Rejestry matrycy sterowania

#define REG_PROFIEXTTABLE_C     (0xE6)				///< Tablica stanu podlaczonych "satelitarnych" modulow mio-1
#define REG_PROFIEXTTABLE_B     (0xE7)				///< Tablica stanu podlaczonych "satelitarnych" modulow mio-1
#define REG_PROFIEXTTABLE_A     (0xF2)				///< Tablica stanu podlaczonych "satelitarnych" modulow mio-1

// Slownik adresow dla rozszerzonej tablicy profibus
#define REG_REFDIC_D			(0xE8)
#define REG_REFDIC_C			(0xE9)
#define REG_REFDIC_B			(0xEA)
#define REG_REFDIC_A			(0xF4)
#define REG_IBO_BAT  			(0xF4)

// Rejestry Diagnostyczne
#define REG_DIAGNOSTIC_6            (0xEB)
#define REG_DIAGNOSTIC_5            (0xEC)
#define REG_DIAGNOSTIC_4            (0xED)
#define REG_DIAGNOSTIC_3            (0xEE)
#define REG_DIAGNOSTIC_2            (0xEF)
#define REG_DIAGNOSTIC_1            (0xF0)
#define REG_DIAGNOSTIC_0            (0xF1)

#define REG_PROFIEXTTABLE_C         (0xE6)				///< Tablica stanu podlaczonych "satelitarnych" modulow mio-1
#define REG_PROFIEXTTABLE_B         (0xE7)				///< Tablica stanu podlaczonych "satelitarnych" modulow mio-1
#define REG_PROFIEXTTABLE_A         (0xF2)				///< Tablica stanu podlaczonych "satelitarnych" modulow mio-1

// Adresy softwarowe can i profi i fizyczny
#define REG_SOFTADDRCAN             (0xF3)

#define REG_EXTPAR_C                (0xF5)

// Rejestr dodatkowych parametrow
#define REG_EXTPAR_B                (0xF6)

// Rejestr zawiera informacje jakie eventy sa aktywne
#define REG_EVENTENABLE             (0xF7)

// Rejestr dodatkowych parametrow
#define REG_EXTPAR_A                (0xF8)

#define REG_SAFESTATE               (0xF9)

// Rejestr informacyjny Reflection profibus
#define REG_PROFIREF                (0xFA)
#define REG_COMMAND                 (0xFB)
#define REG_PCINVAL                 (0xFC)
#define REG_IBO_POWCTRL             (0xFC)
#define REG_ANALOG_CAL              (0xFD)

#define CMD_UNKNOWN                 (0x00)
#define CMD_SOCKETS_SCAN 			(0x01)
#define CMD_SOCKETS_SAVE 			(0x02)
#define CMD_SOCKETS_LOAD 			(0x03)
#define CMD_SOCKETS_TYPE 			(0x04)
#define CMD_REST 					(0x05)
#define CMD_PRID 					(0x06)
#define CMD_CLID 					(0x07)
#define CMD_NAMEA 					(0x08)
#define CMD_NAMEB 					(0x09)
#define CMD_PROFIMUL 				(0x0A)
#define CMD_LOSTTRMS 				(0x0B)
#define CMD_CANFILTER 				(0x0C)
#define CMD_SOCKETS_PROFISETDEBUG  	(0x0D)
#define CMD_SOCKETS_PROFIGETDEBUG   (0x0E)
#define CMD_SOCKETS_PROFIENADEBUG   (0x0F)
#define CMD_EVENTENABLE_GLOBAL   	(0x10)
#define CMD_EVENTENABLE_SLOT	   	(0x11)
#define CMD_PROREFLECTIONTRY	   	(0x12)
#define CMD_REFDIC_A 				(0x13)
#define CMD_REFDIC_B 				(0x14)
#define CMD_SOFTADDR				(0x15)
#define CMD_CANCPUSPD				(0x16)
#define CMD_PERENABLE				(0x17)
#define CMD_SYNCINPERIOD			(0x18)
#define CMD_SYNC_PICCFG			    (0x19)
#define CMD_PCINVAL					(0x1A)
#define CMD_SYNBYTEDELAY			(0x1B)
#define CMD_NORESPONSEWETIME		(0x1C)
#define CMD_SOCKET_CONFIG			(0x1D)
#define CMD_SOCKET_PDICTIONARY	    (0x1E)
#define CMD_DIAG_RESET			    (0x1F)
#define CMD_REFDIC_C 				(0x20)
#define CMD_REFDIC_D 				(0x21)
#define CMD_EVALUATION_VERSION_CNT 	(0x22)

// Flagi konfiguracyjne dla calego urzadzenia
#define PEREN_MASK_ADC 	 		 (0x01)
#define PEREN_MASK_I2C 	 		 (0x02)
#define PEREN_IDLEONWEERROR      (0x80)      // Przechodzi do stanu idle przy dowolnym uszkodzeniu wejscia
#define PEREN_PROFIANALOGTAB     (0x40)		// Jesli prawda do tablicy profibus zostanie doklejona czesc zawierajaca wartosci analogowe
#define PEREN_SYNCFULLTEST       (0x20)		// Jesli prawda to wykonujemy dlugotrwaly test konfiguracji wejsc synchronicznych
#define PEREN_CLRTIMHYSONPROFSYN (0x10)
#define PEREN_STABILITY_TEST_EN  (0x04)		// 1 - Oznacza aktywny test stabilnosci wejsc

// Flagi konfiguracyjne dla pojedynczych gniazd
#define SOCKET_PEREN_PREERROR 	(0x01)		// 1 = Wejscie prezentuje stan uszkodzenia w rejestrze stanu
#define SOCKET_PEREN_PREVALUE 	(0x02)		// 1 = Wejscie prezentuje stan uszkodzenia w rejestrze wartosci
#define SOCKET_PEREN_3STWE 		(0x04)		// 1 = Wejscia 3 stanowe 0 wejscia 2 stanowe
#define SOCKET_ON_HYSTERESE		(0x08)		// 1 = Histereza aktywacji wejscia
#define SOCKET_OFF_HYSTERESE	(0x10)		// 1 = Histereza deaktywacji wejscia

class cMIOCANProtocol
{
private:
    uint8_t dataTable[100];

public:
     cMIOCANProtocol();   // QT constructor
    ~cMIOCANProtocol();

     bool Synchronize( Property* prop );
};

}

#endif // MIOCANPROTOCOL_H
