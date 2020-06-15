TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

DEFINES += NNG_STATIC_LIB

QMAKE_CFLAGS += -Wno-unused-function

SOURCES += pointfile.c
HEADERS += calculation.h

INCLUDEPATH += C:/msys64/mingw64/include
INCLUDEPATH += ..

LIBS += -LC:/msys64/mingw64/lib

LIBS += -lnng
LIBS += -lws2_32 -lmswsock -ladvapi32 -lkernel32 -luser32 -lgdi32 -lwinspool -lshell32 -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32

