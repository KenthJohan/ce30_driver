/*
In 2D computer graphics, a pixel represents a value on a regular grid in two-dimensional space.
In 3D computer graphics, a voxel represents a value on a regular grid in three-dimensional space.
*/


#pragma once
#include <nng/nng.h>
#include <nng/protocol/pair0/pair.h>
#include <nng/supplemental/util/platform.h>
#include <stdio.h>
#include "csc/csc_debug_nng.h"
#include "csc/csc_math.h"
#include "csc/csc_linmat.h"
#include "csc/csc_crossos.h"

#define RGBA(r,g,b,a) (((r) << 0) | ((g) << 8) | ((b) << 16) | ((a) << 24))

#define POINTS_DIM 4

#define LIDAR_W 320
#define LIDAR_H 20
#define LIDAR_WH LIDAR_W*LIDAR_H
#define LIDAR_FPS 30
#define LIDAR_FOV_W 60
#define LIDAR_FOV_H 4
#define LIDAR_INDEX(x,y) ((x)*LIDAR_H + (y))

#define VOXEL_XN 60
#define VOXEL_YN 30
#define VOXEL_ZN 10
#define VOXEL_INDEX(x,y,z) ((z)*VOXEL_XN*VOXEL_YN + (y)*VOXEL_XN + (x))
#define VOXEL_SCALE 0.15f
#define PIXEL_INDEX(x,y) ((y)*VOXEL_XN + (x))



//All socket connection is labeled here:
enum main_nngsock
{
	MAIN_NNGSOCK_POINTCLOUD_POS, //Used for showing raw data from the LIDAR i.e. the pointcloud
	MAIN_NNGSOCK_POINTCLOUD_COL, //Used for showing raw data from the LIDAR i.e. the pointcloud
	MAIN_NNGSOCK_PLANE, //Used for showing the ground plane. The data is 6 vertices of v4f32.
	MAIN_NNGSOCK_TEX, //Used for showing the 2D image of the ground plane.
	MAIN_NNGSOCK_VOXEL, //Used for showing the 3D image of the pointcloud.
	MAIN_NNGSOCK_LINE_POS,
	MAIN_NNGSOCK_LINE_COL,
	MAIN_NNGSOCK_COUNT
};


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




/**
 * @brief main_vox_neighbor
 * @param v 3D array of id
 * @param x Origin coordinate
 * @param y Origin coordinate
 * @param z Origin coordinate
 */
static void main_vox_neighbor (uint8_t *id, uint8_t voxel[], uint8_t x, uint8_t y, uint8_t z)
{
	//This must be true to do an convolution:
	ASSERT (x > 0);
	ASSERT (y > 0);
	ASSERT (z > 0);
	ASSERT (x < (VOXEL_XN-1));
	ASSERT (y < (VOXEL_YN-1));
	ASSERT (z < (VOXEL_ZN-1));

	//(3x3x3) convolution comparison where (x,y,z) is the origin voxel and (a,b,c) is the neighbor voxels:
	for (uint8_t a = x - 1; a <= x + 1; ++a)
	{
		for (uint8_t b = y - 1; b <= y + 1; ++b)
		{
			for (uint8_t c = z - 1; c <= z + 1; ++c)
			{
				//Don't compare it selft:
				if (VOXEL_INDEX(a, b, c) == VOXEL_INDEX(x, y, z)) {continue;}
				//If a neigbor exist then copy the class:
				if (voxel[VOXEL_INDEX(a, b, c)] != 0)
				{
					voxel[VOXEL_INDEX(x, y, z)] = voxel[VOXEL_INDEX(a, b, c)];
					goto loop_break;
				}
			}
		}
	}
loop_break:
	//If no neigbor were found then generate a new classid for the origin voxel:
	if (voxel[VOXEL_INDEX(x, y, z)] == 0)
	{
		(*id)++;
		voxel[VOXEL_INDEX(x, y, z)] = (*id);
	}
}


/**
 * @brief main_test_voxels
 * @param sock Send voxels to GUI client
 * @param voxel 3D array of ids
 * @param p Pointcloud, array of 4D point (x,y,z,w), w is not used yet.
 * @param n Number of points in pointcloud
 */
static void main_test_voxels
(
uint8_t voxel[VOXEL_XN*VOXEL_YN*VOXEL_ZN],
uint8_t img2d[VOXEL_XN*VOXEL_YN],
float const points[],//Stride=4
unsigned points_count
)
{
	//Reset each voxel:
	memset (voxel, 0, VOXEL_XN*VOXEL_YN*VOXEL_ZN);
	//Reset each pixel:
	memset (img2d, 0, VOXEL_XN*VOXEL_YN*sizeof(uint32_t));
	//Each voxel will be given an ID:
	uint8_t id = 0;

	//Iterate each point in pointcloud:
	for (unsigned i = 0; i < points_count; ++i, points+=4)
	{
		//Map 3d points to a index in the 3D array:
		float fx = (points[0])/VOXEL_SCALE;//Downscale the LIDAR points to lower resolution.
		float fy = (points[1])/VOXEL_SCALE;//Downscale the LIDAR points to lower resolution.
		float fz = (points[2])/VOXEL_SCALE;//Downscale the LIDAR points to lower resolution.
		uint8_t ux = fx; //This will be the direction the LIDAR is pointing at, (fx) will never be negative.
		uint8_t uy = fy+VOXEL_YN/2; //LIDAR (fy)=0 coordinate is moved to middle of the 3D image.
		uint8_t uz = fz+VOXEL_ZN/2; //LIDAR (fz)=0 coordinate is moved to middle of the 3D image.
		//Ignore edges because those can not be proccessed with convolution:
		if (ux >= (VOXEL_XN-1)){continue;}
		if (uy >= (VOXEL_YN-1)){continue;}
		if (uz >= (VOXEL_ZN-1)){continue;}
		if (ux <= 0){continue;}
		if (uy <= 0){continue;}
		if (uz <= 0){continue;}
		//if (voxel1[VOX_I(ux, uy, uz)]){continue;}
		//Project point to the 2D image, the pixel value represent (uz):
		//img2d[PIXEL_INDEX(ux, uy)] = (0xFF << 0) | (uz << 8) | (0xFF << 24);
		img2d[PIXEL_INDEX(ux, uy)] = img2d[PIXEL_INDEX(ux, uy)]/2 + uz*4;
		//printf ("%x\n", img2d[PIXEL_INDEX(ux, uy)]);
		main_vox_neighbor (&id, voxel, ux, uy, uz);
	}
}



struct gobj_line
{
	nng_socket sock;
	uint32_t cap;
	uint32_t last;
	float * lines;//Stride=4
};

void gobj_line_push (struct gobj_line * obj, float x, float y, float z)
{
	float * lines = obj->lines + obj->last * 4;
	lines[0] = x;
	lines[1] = y;
	lines[2] = z;
	lines[3] = 1.0f;
	obj->last++;
}


void gobj_line_send (struct gobj_line * obj)
{
	int r;
	r = nng_send (obj->sock, obj->lines, obj->last*4*sizeof(float), 0);
	if (r)
	{
		perror (nng_strerror (r));
	}
}

