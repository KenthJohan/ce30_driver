TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

QMAKE_CFLAGS += -Wno-unused-function

SOURCES += test_LAPACKE_ssyev.c

INCLUDEPATH += C:/msys64/mingw64/include
INCLUDEPATH += ..
HEADERS += ../csc/csc_math.h

LIBS += -LC:/msys64/mingw64/lib

LIBS += -llapacke -lblas -lm
