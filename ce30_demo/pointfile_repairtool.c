#include <float.h>
#include <unistd.h>
#include <stdio.h>

#include <nng/nng.h>
#include <nng/protocol/pair0/pair.h>
#include <nng/supplemental/util/platform.h>

#include "csc/csc_debug_nng.h"
#include "csc/csc_math.h"
#include "csc/csc_linmat.h"
#include "csc/csc_crossos.h"
#include "csc/csc_malloc_file.h"

#include "calculation.h"


void points_read (char const s[], float p[], uint32_t *n)
{
	uint32_t i = 0;
	float v[4] = {0.0f};
	while (s[0] != '\0')
	{
		char * e;//Used for endptr of float token
		v[0] = strtof (s, &e);//Convert string to float starting from (s)
		if (e == s) {s++; continue;}//If parse fails then try again
		s = e;//Parse success goto to next token
		v[1] = strtof (s, &e);//Convert string to float starting from (s)
		if (e == s) {s++; continue;}//If parse fails then try again
		s = e;//Parse success goto to next token
		v[2] = strtof (s, &e);//Convert string to float starting from (s)
		if (e == s) {s++; continue;}//If parse fails then try again
		s = e;//Parse success goto to next token
		memcpy (p, v, sizeof(v));//If a entire point (v) got successfully parsed then copy this point into the point array (p)
		p += 4;//The point array (p) consist of 4 dim points
		i++;//Keep track of how many points got parsed
	}
	(*n) = i;
}


void points_print (float p[], uint32_t n)
{
	for (uint32_t i = 0; i < n; ++i)
	{
		printf ("%f %f %f\n", p[0], p[1], p[2]);
		p += 4;
	}
}




#define POINTS_DIM 4


void point_select (uint32_t pointcol[LIDAR_WH], int x, int y, uint32_t color)
{
	int index = LIDAR_INDEX(x,y);
	ASSERT (index < LIDAR_WH);
	printf ("index %i\n", index);
	pointcol[index] = color;
}


int main()
{
	csc_crossos_enable_ansi_color();
	char const * txtpoint = csc_malloc_file ("../ce30_demo/txtpoints/14_14_02_29138.txt");
	float pointpos[LIDAR_WH*POINTS_DIM] = {0.0f};
	uint32_t pointcol[LIDAR_WH] = {0};
	for (int i = 0; i < LIDAR_WH; ++i) {pointcol[i] = RGBA (0xFF, 0xFF, 0xFF, 0xFF);}

	uint32_t n = LIDAR_WH;
	points_read (txtpoint, pointpos, &n);
	//points_print (point, n);

	nng_socket socks[MAIN_NNGSOCK_COUNT] = {{0}};
	main_nng_pairdial (socks + MAIN_NNGSOCK_POINTCLOUD,       "tcp://192.168.1.176:9002");
	main_nng_pairdial (socks + MAIN_NNGSOCK_POINTCLOUD_COLOR, "tcp://192.168.1.176:9003");
	main_nng_pairdial (socks + MAIN_NNGSOCK_TEX,              "tcp://192.168.1.176:9004");
	main_nng_pairdial (socks + MAIN_NNGSOCK_VOXEL,            "tcp://192.168.1.176:9005");

	main_nng_send (socks[MAIN_NNGSOCK_POINTCLOUD], pointpos, LIDAR_W*LIDAR_H*4*sizeof(float));
	main_nng_send (socks[MAIN_NNGSOCK_POINTCLOUD_COLOR], pointcol, LIDAR_W*LIDAR_H*sizeof(uint32_t));


	int x = 0;
	int y = 0;
	while (1)
	{
		int c = getchar();
		int d = 0;
		point_select (pointcol, x, y, RGBA (0xFF, 0xFF, 0xFF, 0xFF));
		switch (c)
		{
		case 'q':
			return 0;

		case 'a':
			scanf("%d", &d);
			x -= d;
			break;
		case 'd':
			scanf("%d", &d);
			x += d;
			break;
		case 's':
			scanf("%d", &d);
			y -= d;
			break;
		case 'w':
			scanf("%d", &d);
			y += d;
			break;
		case 'r':
			free (txtpoint);
			txtpoint = csc_malloc_file ("../ce30_demo/txtpoints/14_14_02_29138.txt");
			points_read (txtpoint, pointpos, &n);
			main_nng_send (socks[MAIN_NNGSOCK_POINTCLOUD], pointpos, LIDAR_W*LIDAR_H*4*sizeof(float));
			break;
		}
		if (c)
		{
			printf ("c:%i, xy=(%i %i)\n", c, x, y);
			point_select (pointcol, x, y, RGBA (0xFF, 0x00, 0x00, 0xFF));
			main_nng_send (socks[MAIN_NNGSOCK_POINTCLOUD_COLOR], pointcol, LIDAR_W*LIDAR_H*sizeof(uint32_t));
		}

	}

	return 0;

/*
	float point1[LIDAR_WH*POINTS_DIM] = {0.0f};
	for (int i = 151; i < LIDAR_W; ++i)
	{
		printf ("Column %i\n", i);
		memcpy (point1, point+i*LIDAR_H*POINTS_DIM, 1*LIDAR_H*POINTS_DIM*sizeof(float));
		main_nng_send (socks[MAIN_NNGSOCK_POINTCLOUD], point1, LIDAR_W*LIDAR_H*4*sizeof(float));
		usleep(1000*200);
		//break;
	}


	for (uint32_t i = 0; i < LIDAR_W; ++i)
	{
		float point1[POINTS_CAP*POINTS_DIM] = {0.0f};
		memcpy (point1, point + i*LIDAR_H*POINTS_DIM, LIDAR_H*POINTS_DIM*sizeof(float));
		main_nng_send (socks[MAIN_NNGSOCK_POINTCLOUD], point1, LIDAR_W*LIDAR_H*4*sizeof(float));
		usleep(1000*200);
	}
	return 0;

	float points_z[LIDAR_W*LIDAR_H];
	for (uint32_t i = 0; i < LIDAR_W*LIDAR_H; ++i)
	{
		points_z[i] = point[i*POINTS_DIM + 2];
	}

	float point1[POINTS_CAP*POINTS_DIM] = {0.0f};
	for (uint32_t i = 0; i < LIDAR_W; ++i)
	{
		point1[i*POINTS_DIM + 0] = 0.0f;
		point1[i*POINTS_DIM + 1] = (float)i*0.01f;
		point1[i*POINTS_DIM + 2] = vf32_sum (LIDAR_H, points_z + i*LIDAR_H*POINTS_DIM);
	}
	main_nng_send (socks[MAIN_NNGSOCK_POINTCLOUD], point1, LIDAR_W*LIDAR_H*4*sizeof(float));
	*/

}
