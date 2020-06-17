#include <float.h>
#include <unistd.h>
#include <stdio.h>

#include <nng/nng.h>
#include <nng/protocol/pair0/pair.h>
#include <nng/supplemental/util/platform.h>

#include <lapacke.h>

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

struct point_xyzw
{
	float x;
	float y;
	float z;
	float w;
};


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
	struct point_xyzw point1[LIDAR_WH];
	uint32_t pointcol[LIDAR_WH] = {0};
	for (int i = 0; i < LIDAR_WH; ++i) {pointcol[i] = RGBA (0xFF, 0xFF, 0xFF, 0xFF);}

	uint32_t n = LIDAR_WH;
	points_read (txtpoint, pointpos, &n);
	memcpy (point1, pointpos, n*4*sizeof(float));
	//points_print (point, n);

	nng_socket socks[MAIN_NNGSOCK_COUNT] = {{0}};
	main_nng_pairdial (socks + MAIN_NNGSOCK_POINTCLOUD,       "tcp://192.168.1.176:9002");
	main_nng_pairdial (socks + MAIN_NNGSOCK_POINTCLOUD_COLOR, "tcp://192.168.1.176:9003");
	main_nng_pairdial (socks + MAIN_NNGSOCK_TEX,              "tcp://192.168.1.176:9004");
	main_nng_pairdial (socks + MAIN_NNGSOCK_VOXEL,            "tcp://192.168.1.176:9005");
	main_nng_pairdial (socks + MAIN_NNGSOCK_LINES,            "tcp://192.168.1.176:9006");

	main_nng_send (socks[MAIN_NNGSOCK_POINTCLOUD], pointpos, LIDAR_WH*4*sizeof(float));
	main_nng_send (socks[MAIN_NNGSOCK_POINTCLOUD_COLOR], pointcol, LIDAR_WH*sizeof(uint32_t));


	/*
	float rotx[4*4];
	m4f32_identity (rotx);
	M4_ROTY (rotx, 80.0f*(M_PI/180.0f));
	for (uint32_t i = 0; i < LIDAR_WH; ++i)
	{
		mv4f32_mul ((float*)(point1 + i), rotx, (float*)(point1 + i));
		//point1[i].z += 1.0f;
	}
	*/

	float c[3*3];
	float mean[4];
	m3f32_coveriance (c, mean, (float*)point1, LIDAR_WH);
	m3f32_print (c, stdout);

	float w[3];
	LAPACKE_ssyev (LAPACK_COL_MAJOR, 'V', 'U', 3, c, 3, w);
	m3f32_print (c, stdout);

	float lines[12*4] =
	{
	/*
	0.0f, 0.0f, 0.0f, 1.0f,
	1.0f, 0.0f, 0.0f, 1.0f,

	0.0f, 0.0f, 0.0f, 1.0f,
	0.0f, 1.0f, 0.0f, 1.0f,

	0.0f, 0.0f, 0.0f, 1.0f,
	0.0f, 0.0f, 1.0f, 1.0f,
	*/
	0.0f, 0.0f, 0.0f, 1.0f,
	c[0], c[1], c[2], 1.0f,
	0.0f, 0.0f, 0.0f, 1.0f,
	-c[0], -c[1], -c[2], 1.0f,

	0.0f, 0.0f, 0.0f, 1.0f,
	c[3], c[4], c[5], 1.0f,
	0.0f, 0.0f, 0.0f, 1.0f,
	-c[3], -c[4], -c[5], 1.0f,

	0.0f, 0.0f, 0.0f, 1.0f,
	c[6], c[7], c[8], 1.0f,
	0.0f, 0.0f, 0.0f, 1.0f,
	-c[6], -c[7], -c[8], 1.0f,

	};
	{
		int r;
		r = nng_send (socks[MAIN_NNGSOCK_LINES], lines, 12*4*sizeof(float), 0);
		perror (nng_strerror (r));
	}


	for (uint32_t i = 0; i < LIDAR_WH; ++i)
	{
		vvf32_sub (4, (float*)(point1 + i), (float*)(point1 + i), mean);
	}

	/*
	float rotx[4*4];
	m4f32_identity (rotx);
	M4_ROTY (rotx, 20.0f*(M_PI/180.0f));
	for (uint32_t i = 0; i < LIDAR_WH; ++i)
	{
		mv4f32_mul ((float*)(point1 + i), rotx, (float*)(point1 + i));
		//point1[i].z += 1.0f;
	}
	*/

	/*
	float points_z[LIDAR_WH];
	for (uint32_t i = 0; i < LIDAR_WH; ++i)
	{
		points_z[i] = pointpos[i*POINTS_DIM + 2];
	}
	ASSERT (sizeof (point1) == LIDAR_WH*POINTS_DIM*sizeof (float));
	memset (point1, 0, sizeof (point1));
	memcpy (point1, pointpos, sizeof (point1));

	for (uint32_t i = 0; i < LIDAR_W; ++i)
	{
		point1[LIDAR_INDEX(i, 0)].z = vf32_sum (LIDAR_H, points_z + i*LIDAR_H) + 1.0f;
		pointcol[LIDAR_INDEX(i, 0)] = RGBA (0xFF, 0x00, 0x00, 0xFF);
	}
	//for (int i = 0; i < LIDAR_WH; ++i) {pointcol[i] = RGBA (0xFF, 0x00, 0xFF, 0xFF);}
	*/
	{
		int r;
		r = nng_send (socks[MAIN_NNGSOCK_POINTCLOUD], point1, LIDAR_W*LIDAR_H*4*sizeof(float), 0);
		perror (nng_strerror (r));
		r = nng_send (socks[MAIN_NNGSOCK_POINTCLOUD_COLOR], pointcol, LIDAR_W*LIDAR_H*sizeof(uint32_t), 0);
		perror (nng_strerror (r));
	}

}
