#include <iostream>
#include <float.h>
#include <ce30_driver/ce30_driver.h>

#include <nng/nng.h>
#include <nng/protocol/pair0/pair.h>
#include <nng/supplemental/util/platform.h>

#include "csc/csc_debug_nng.h"
#include "csc/csc_math.h"
#include "csc/csc_linmat.h"
#include "csc/csc_crossos.h"

using namespace std;
using namespace ce30_driver;

#define POINTC_W 320
#define POINTC_H 20


static void main_nng_send (nng_socket socket, void * data, unsigned size8)
{
	int r;
	r = nng_send (socket, data, size8, NNG_FLAG_NONBLOCK);
	if (r == 0)
	{
		return;
	}
	else if (r == NNG_EAGAIN)
	{
		return;
	}
	else if (r == NNG_ECLOSED)
	{
		printf ("NNG_ECLOSED\n");
		return;
	}
}


static void main_nng_pairdial (nng_socket * sock, char const * address)
{
	int r;
	r = nng_pair0_open (sock);
	NNG_EXIT_ON_ERROR (r);
	r = nng_dial (*sock, address, NULL, 0);
	NNG_EXIT_ON_ERROR (r);
}


static void random_points (float v[], unsigned n)
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


static void convert_lidar_to_v4f32_array (float points[], Scan const &scan)
{
	ASSERT (POINTC_W == scan.Width());
	ASSERT (POINTC_H == scan.Height());
	float * p = points;
	for (int x = 0; x < POINTC_W; ++x)
	{
		for (int y = 0; y < POINTC_H; ++y)
		{
			Channel channel = scan.at(x, y);
			p[0] = channel.point().x;
			p[1] = channel.point().y;
			p[2] = channel.point().z;
			p[3] = 1.0f;
			p += 4;
		}
	}
}



#define VOX_XN 60
#define VOX_YN 30
#define VOX_ZN 10
#define VOX_I(x,y,z) ((z)*VOX_XN*VOX_YN + (y)*VOX_XN + (x))
#define VOX_SCALE 0.1f


/**
 * @brief main_vox_neighbor
 * @param v 3D array of id
 * @param x Origin coordinate
 * @param y Origin coordinate
 * @param z Origin coordinate
 */
static void main_vox_neighbor (uint8_t v[], uint8_t x, uint8_t y, uint8_t z)
{
	ASSERT (x > 0);
	ASSERT (y > 0);
	ASSERT (z > 0);
	ASSERT (x < (VOX_XN-1));
	ASSERT (y < (VOX_YN-1));
	ASSERT (z < (VOX_ZN-1));
	static uint8_t id = 0;

	//(3x3x3) convolution comparison where (x,y,z) is the origin and (a,b,c) is the neighbors:
	for (uint8_t a = x - 1; a <= x + 1; ++a)
	{
		for (uint8_t b = y - 1; b <= y + 1; ++b)
		{
			for (uint8_t c = z - 1; c <= z + 1; ++c)
			{
				//Don't compare it selft:
				if (VOX_I(a, b, c) == VOX_I(x, y, z)) {continue;}
				//If neigbor is classified then copy the class:
				if (v[VOX_I(a, b, c)] != 0)
				{
					v[VOX_I(x, y, z)] = v[VOX_I(a, b, c)];
					goto loop_break;
				}
			}
		}
	}
loop_break:
	//If no neigbor had any class then generate a new one:
	if (v[VOX_I(x, y, z)] == 0)
	{
		id++;
		v[VOX_I(x, y, z)] = id;
	}
}


/**
 * @brief main_test_voxels
 * @param sock Send voxels to GUI client
 * @param voxel 3D array of ids
 * @param p Pointcloud, array of 4D point (x,y,z,w), w is not used yet.
 * @param n Number of points in pointcloud
 */
static void main_test_voxels (nng_socket sock, uint8_t voxel[VOX_XN*VOX_YN*VOX_ZN], float const p[], unsigned n)
{
	//Reset each voxel:
	memset (voxel, 0, VOX_XN*VOX_YN*VOX_ZN);

	//Iterate each point in pointcloud:
	for (unsigned i = 0; i < n; ++i, p+=4)
	{
		//Map 3d points to a index in the 3D array:
		float fx = (p[0])/VOX_SCALE;
		float fy = (p[1])/VOX_SCALE;
		float fz = (p[2])/VOX_SCALE;
		uint8_t ux = fx;
		uint8_t uy = fy+VOX_YN/2;
		uint8_t uz = fz+VOX_ZN/2;
		//Do not proccess edges because those can not be compared with convolution:
		if (ux >= (VOX_XN-1)){continue;}
		if (uy >= (VOX_YN-1)){continue;}
		if (uz >= (VOX_ZN-1)){continue;}
		if (ux <= 0){continue;}
		if (uy <= 0){continue;}
		if (uz <= 0){continue;}
		main_vox_neighbor (voxel, ux, uy, uz);
	}
	main_nng_send (sock, voxel, VOX_XN*VOX_YN*VOX_ZN);
}





enum main_nngsock
{
	MAIN_NNGSOCK_POINTCLOUD,
	MAIN_NNGSOCK_PLANE,
	MAIN_NNGSOCK_TEX,
	MAIN_NNGSOCK_VOXEL,
	MAIN_NNGSOCK_COUNT
};


int main()
{
	csc_crossos_enable_ansi_color();

	nng_socket socks[MAIN_NNGSOCK_COUNT] = {{0}};
	main_nng_pairdial (socks + MAIN_NNGSOCK_POINTCLOUD, "tcp://192.168.1.176:9002");
	main_nng_pairdial (socks + MAIN_NNGSOCK_VOXEL, "tcp://192.168.1.176:9005");

	float points[POINTC_W*POINTC_H*4] = {0.0f};
	uint8_t voxel[VOX_XN*VOX_YN*VOX_ZN] = {0};

	random_points (points, POINTC_W*POINTC_H);
	main_nng_send (socks[MAIN_NNGSOCK_POINTCLOUD], points, POINTC_W*POINTC_H*4*sizeof(float));

	UDPSocket socket;
	if (socket.Connect() != Diagnose::connect_successful)
	{
		return -1;
	}
	Packet packet;
	Scan scan;
	printf ("Loop:\n");
	while (true)
	{
		if (!GetPacket (packet, socket)){continue;}
		unique_ptr<ParsedPacket> parsed = packet.Parse();
		if (!parsed){continue;}
		scan.AddColumnsFromPacket (*parsed);
		if (!scan.Ready()){continue;}
		convert_lidar_to_v4f32_array (points, scan);
		scan.Reset();

		main_nng_send (socks[MAIN_NNGSOCK_POINTCLOUD], points, POINTC_W*POINTC_H*4*sizeof(float));
		main_test_voxels (socks[MAIN_NNGSOCK_VOXEL], voxel, points, POINTC_W*POINTC_H);
	}
}
