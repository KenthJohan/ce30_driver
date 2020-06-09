TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

#DEFINES += UDP_SOCKET_USE_BOOST_API
DEFINES += UDP_SOCKET_USE_WINSOCK2_API
DEFINES += __USE_MINGW_ANSI_STDIO=1

QMAKE_CFLAGS += -Wno-unused-function

SOURCES += ../ce30_driver/utils.cpp
SOURCES += ../ce30_driver/udp_socket.cpp
SOURCES += ../ce30_driver/udp_socket_win_impl.cpp
SOURCES += ../ce30_driver/timed_udp_socket.cpp
SOURCES += ../ce30_driver/data_types.cpp
SOURCES += ../ce30_driver/packet.cpp
SOURCES += ../ce30_driver/udp_server.cpp
SOURCES += ce30_demo2.cpp

INCLUDEPATH += C:/msys64/mingw64/include
INCLUDEPATH += ..

LIBS += -LC:/msys64/mingw64/lib
LIBS += -lboost_system-mt -lm -lWs2_32
