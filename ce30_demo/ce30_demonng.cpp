#include <iostream>
#include <ce30_driver/ce30_driver.h>

#include <nng/nng.h>
#include <nng/protocol/pair0/pair.h>
#include <nng/supplemental/util/platform.h>

#include "../csc/csc_debug_nng.h"

using namespace std;
using namespace ce30_driver;

#define POINTC_W 320
#define POINTC_H 20


void random_points (float v[], unsigned n)
{
	while (n--)
	{
		v[0] = (float)rand() / (float)RAND_MAX;
		v[1] = (float)rand() / (float)RAND_MAX;
		v[2] = ((float)rand() / (float)RAND_MAX) * 10.0f;
		v[3] = 1.0f;
		v += 4;
	}
}

void send123 (nng_socket socket, float v[], unsigned n)
{
	int r;
	r = nng_send (socket, v, sizeof (float) * n * 4, NNG_FLAG_NONBLOCK);
	if (r != NNG_EAGAIN)
	{
		NNG_EXIT_ON_ERROR (r);
	}
}


int main()
{

	nng_socket socket_draw;


	{
		int r;
		r = nng_pair0_open (&socket_draw);
		NNG_EXIT_ON_ERROR (r);
		r = nng_dial (socket_draw, "tcp://192.168.1.176:9002", NULL, 0);
		NNG_EXIT_ON_ERROR (r);
	}

	float points[POINTC_W*POINTC_H*4] = {0.0f};
	float points2[POINTC_W*POINTC_H*4] = {0.0f};
	random_points (points, POINTC_W*POINTC_H);
	send123 (socket_draw, points, POINTC_W*POINTC_H);

	UDPSocket socket;
	if (socket.Connect() != Diagnose::connect_successful)
	{
		return -1;
	}
	Packet packet;
	Scan scan;
	while (true)
	{
		if (!GetPacket (packet, socket)){continue;}
		unique_ptr<ParsedPacket> parsed = packet.Parse();
		if (!parsed){continue;}
		scan.AddColumnsFromPacket (*parsed);
		if (!scan.Ready()){continue;}
		//printf ("wh: %03i %03i\n", scan.Width(), scan.Height());

		//y:=y+h[k]*x

		float * p = points;
		for (int x = 0; x < scan.Width(); ++x)
		{
			for (int y = 0; y < scan.Height(); ++y)
			{
				Channel channel = scan.at(x, y);
				p[0] = channel.point().x*10.0f;
				p[2] = channel.point().y*10.0f;
				p[1] = -channel.point().z*10.0f;
				p[3] = 1.0f;
				p += 4;
			}
		}

		for (unsigned i = 0; i < POINTC_W*POINTC_H*4; ++i)
		{
			float k = 0.01f;
			points2[i] = ((1.0f - k) * points2[i]) + (k * points[i]);
		}


		scan.Reset();
		send123 (socket_draw, points2, POINTC_W*POINTC_H);
	}
}
