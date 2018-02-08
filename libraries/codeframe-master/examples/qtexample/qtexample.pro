#-------------------------------------------------
#
# Project created by QtCreator 2017-09-08T19:45:36
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

win32 { CONFIG-=debug_and_release }

CONFIG(debug, debug|release) {
    DESTDIR = $$OUT_PWD/../
    OBJECTS_DIR = build/obj_d
    MOC_DIR = build/moc_d
    UI_DIR =  build/ui_d
    RCC_DIR = build/rcc_d
    unix:TARGET = qtexample_d
    win32:TARGET = qtexample_d
} else {
    DESTDIR = $$OUT_PWD/../
    OBJECTS_DIR = build/obj_r
    MOC_DIR = build/moc_r
    UI_DIR =  build/ui_r
    RCC_DIR = build/rcc_r
    unix:TARGET = qtexample
    win32:TARGET = qtexample
}

TEMPLATE = app

SOURCES += main.cpp\
        mainwindow.cpp \
    objectcontainer.cpp \
    objectchild.cpp

HEADERS  += mainwindow.h \
    objectcontainer.h \
    objectchild.h

FORMS    += mainwindow.ui

INCLUDEPATH += $$PWD/../libraries/sigslot-1.0.0
INCLUDEPATH += $$PWD/../libraries/smartpointer
INCLUDEPATH += $$PWD/../qtwidgets
INCLUDEPATH += $$PWD/../utilities
INCLUDEPATH += $$PWD/../src

LIBS += $$OUT_PWD/../libqtwidgets.a
LIBS += $$OUT_PWD/../libutilities.a
LIBS += $$OUT_PWD/../libserializable.a

PRE_TARGETDEPS += $$OUT_PWD/../libqtwidgets.a
PRE_TARGETDEPS += $$OUT_PWD/../libutilities.a
PRE_TARGETDEPS += $$OUT_PWD/../libserializable.a

RESOURCES += \
    qtexample.qrc
