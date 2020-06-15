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

#include "calculation.h"

using namespace std;
using namespace ce30_driver;




static void convert_lidar_to_v4f32_array (float p[], Scan const &scan)
{
	ASSERT (LIDAR_W == scan.Width());
	ASSERT (LIDAR_H == scan.Height());
	for (int x = 0; x < LIDAR_W; ++x)
	{
		for (int y = 0; y < LIDAR_H; ++y)
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



int main()
{
	csc_crossos_enable_ansi_color();

	nng_socket socks[MAIN_NNGSOCK_COUNT] = {{0}};
	main_nng_pairdial (socks + MAIN_NNGSOCK_POINTCLOUD, "tcp://192.168.1.176:9002");
	main_nng_pairdial (socks + MAIN_NNGSOCK_TEX,        "tcp://192.168.1.176:9004");
	main_nng_pairdial (socks + MAIN_NNGSOCK_VOXEL,      "tcp://192.168.1.176:9005");

	float points[LIDAR_W*LIDAR_H*4] = {0.0f};
	uint8_t voxel[VOXEL_XN*VOXEL_YN*VOXEL_ZN] = {0};
	uint8_t pixel[VOXEL_XN*VOXEL_YN] = {0};
	uint32_t pixel_rgba[VOXEL_XN*VOXEL_YN] = {0};

	random_points (points, LIDAR_W*LIDAR_H);
	main_nng_send (socks[MAIN_NNGSOCK_POINTCLOUD], points, LIDAR_W*LIDAR_H*4*sizeof(float));

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

		main_test_voxels (voxel, pixel, points, LIDAR_W*LIDAR_H);


		main_nng_send (socks[MAIN_NNGSOCK_POINTCLOUD], points, LIDAR_W*LIDAR_H*4*sizeof(float));
		//Send the 3D image to the connected socket:
		main_nng_send (socks[MAIN_NNGSOCK_VOXEL], voxel, VOXEL_XN*VOXEL_YN*VOXEL_ZN);
		//Send the 2D image to the connected socket:
		main_nng_send (socks[MAIN_NNGSOCK_TEX], pixel_rgba, VOXEL_XN*VOXEL_YN*sizeof(uint32_t));

		for (uint32_t i = 0; i < VOXEL_XN*VOXEL_YN; ++i)
		{
			pixel_rgba[i] = RGBA (pixel[i], pixel[i], pixel[i], pixel[i] ? 0xFF : 0x00);
		}

	}
}
