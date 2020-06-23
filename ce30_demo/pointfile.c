#include <float.h>
#include <unistd.h>
#include <stdio.h>

#include <nng/nng.h>
#include <nng/protocol/pair0/pair.h>
#include <nng/supplemental/util/platform.h>

#include <lapacke.h>
#include <OpenBLAS/cblas.h>

#include "csc/csc_debug_nng.h"
#include "csc/csc_crossos.h"
#include "csc/csc_malloc_file.h"

#include "csc/csc_math.h"
#include "csc/csc_linmat.h"
#include "csc/csc_m3f32.h"
#include "csc/csc_m4f32.h"
#include "csc/csc_v3f32.h"
#include "csc/csc_v4f32.h"
#include "csc/csc_qf32.h"

#include "calculation.h"


#define IMG_XN 10
#define IMG_YN 100


void point_select (uint32_t pointcol[LIDAR_WH], int x, int y, uint32_t color)
{
	int index = LIDAR_INDEX(x,y);
	ASSERT (index < LIDAR_WH);
	printf ("index %i\n", index);
	pointcol[index] = color;
}




void pca (float v[], uint32_t n, float c[3*3], float w[3])
{
	//3 dimensions:
	uint32_t const dim = 3;
	//Memory layout, column vector every 4 float:
	uint32_t const stride = 4;
	float mean[4] = {0};
	memset (c, 0, sizeof (float)*dim*dim);
	memset (mean, 0, sizeof (float)*dim);
	//Calculate the (mean) coordinate from (v):
	vf32_addv (dim, mean, 0, mean, 0, v, stride, n);
	vsf32_mul (dim, mean, mean, 1.0f / (float)n);
	//Move all (v) points to origin using coordinate (mean):
	vf32_subv (dim, v, stride, v, stride, mean, 0, n);
	//Calculate the covariance matrix (c) from (v):
	m3f32_symmetric_xxt (c, v, stride, n);
	vsf32_mul (dim*dim, c, c, 1.0f / ((float)n - 1.0f));
	mf32_print (c, dim, dim, stdout);
	//Calculate the eigen vectors (c) and eigen values (w) from covariance matrix (c):
	LAPACKE_ssyev (LAPACK_COL_MAJOR, 'V', 'U', dim, c, dim, w);
	mf32_print (c, dim, dim, stdout);
}


void selective_cpy (float dst[], uint32_t dst_stride, float const src[], uint32_t src_stride, uint32_t *n, uint32_t dim, float k2)
{
	uint32_t j = 0;
	for (uint32_t i = 0; i < (*n); ++i)
	{
		float l2 = vvf32_dot (dim, src, src);
		if (l2 > k2)
		{
			vf32_cpy (dim, dst, src);
			dst += dst_stride;
			j++;
		}
		src += src_stride;
	}
	(*n) = j;
}


void pointfall (float pix[], uint32_t w, uint32_t h, float v[], uint32_t v_stride, uint32_t x_count)
{
	for (uint32_t i = 0; i < x_count; ++i, v += v_stride)
	{
		float x = v[0]*20.0f + w/2.0f;
		float y = v[1]*20.0f + h/2.0f;
		if (x >= w){continue;}
		if (y >= h){continue;}
		if (x < 0){continue;}
		if (y < 0){continue;}
		float z = v[2];
		uint32_t index = (y * w) + x;
		pix[index] += (z > 0.0f) ? 10.0f : 0.0f;
	}
}





int main()
{
	csc_crossos_enable_ansi_color();
	char const * txtpoint = csc_malloc_file ("../ce30_demo/txtpoints/14_14_02_29138.txt");

	float point_pos0[LIDAR_WH*POINTS_DIM] = {0.0f};
	float point_pos1[LIDAR_WH*POINTS_DIM] = {0.0f};
	uint32_t point_pos1_count = LIDAR_WH;

	uint32_t pointcol[LIDAR_WH] = {0};
	for (int i = 0; i < LIDAR_WH; ++i) {pointcol[i] = RGBA (0xFF, 0xFF, 0xFF, 0xFF);}

	points_read (txtpoint, point_pos0, &point_pos1_count);
	selective_cpy (point_pos1, POINTS_DIM, point_pos0, POINTS_DIM, &point_pos1_count, 3, 1.0f);
	//points_print (point, n);

	nng_socket socks[MAIN_NNGSOCK_COUNT] = {{0}};
	main_nng_pairdial (socks + MAIN_NNGSOCK_POINTCLOUD_POS,       "tcp://192.168.1.176:9002");
	main_nng_pairdial (socks + MAIN_NNGSOCK_POINTCLOUD_COL, "tcp://192.168.1.176:9003");
	main_nng_pairdial (socks + MAIN_NNGSOCK_TEX,              "tcp://192.168.1.176:9004");
	main_nng_pairdial (socks + MAIN_NNGSOCK_VOXEL,            "tcp://192.168.1.176:9005");
	main_nng_pairdial (socks + MAIN_NNGSOCK_LINE_POS,            "tcp://192.168.1.176:9006");
	main_nng_pairdial (socks + MAIN_NNGSOCK_LINE_COL,      "tcp://192.168.1.176:9007");

	main_nng_send (socks[MAIN_NNGSOCK_POINTCLOUD_POS], point_pos0, LIDAR_WH*4*sizeof(float));
	main_nng_send (socks[MAIN_NNGSOCK_POINTCLOUD_COL], pointcol, LIDAR_WH*sizeof(uint32_t));


	/*
	for (uint32_t i = 0; i < point_pos1_count; ++i)
	{
		printf ("%f %f %f\n", point_pos1[i*POINTS_DIM + 0], point_pos1[i*POINTS_DIM + 1], point_pos1[i*POINTS_DIM + 2]);
	}
	*/

	float c[4*4];
	float w[4];
	pca ((float*)point_pos1, point_pos1_count, c, w);
	//printf ("%f %f %f\n", w[0], w[1], w[2]);
	float rotation[4*4] =
	{
	c[3], c[6], c[0], 0.0f,
	c[4], c[7], c[1], 0.0f,
	c[5], c[8], c[2], 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f
	};
	for (uint32_t i = 0; i < point_pos1_count; ++i)
	{
		//point_pos1[i*POINTS_DIM + 3] = 1.0f;
		float * v = point_pos1 + i*POINTS_DIM;
		mv4f32_mul (v, rotation, v);
		v[0] += 0.0f;
		//vsf32_mul (4, v, v, 2.0f);
	}
	//cblas_sgemm (CblasColMajor, CblasTrans, CblasNoTrans, 4, point_pos1_count, 4, 1.0f, rotation, 4, point_pos1, 4, 0.0f, point_pos1, 4);
	for (uint32_t i = 0; i < point_pos1_count; ++i)
	{
		float x = point_pos1[i*POINTS_DIM + 0];
		float y = point_pos1[i*POINTS_DIM + 1];
		float z = point_pos1[i*POINTS_DIM + 2];
		if (x < 0.0f) {continue;}
		if (y < 0.0f) {continue;}
		printf ("%f %f %f\n", x, y, z);
	}
	float pix[IMG_XN*IMG_YN] = {0};
	pointfall (pix, IMG_XN, IMG_YN, point_pos1, 4, point_pos1_count);
	uint32_t pix_rgba[IMG_XN*IMG_YN];
	for (uint32_t i = 0; i < IMG_XN*IMG_YN; ++i)
	{
		uint8_t r = CLAMP (pix[i], 0.0f, 255.0f);
		pix_rgba[i] = RGBA (r, 0x00, 0x00, 0xFF);
	}
	pix_rgba[0*IMG_XN + 0] |= RGBA(0x00, 0xFF, 0x00, 0xFF);
	pix_rgba[0*IMG_XN + 1] |= RGBA(0x00, 0xFF, 0x00, 0xFF);
	pix_rgba[2*IMG_XN + 0] |= RGBA(0x00, 0xFF, 0xff, 0xFF);
	pix_rgba[2*IMG_XN + 1] |= RGBA(0x00, 0xFF, 0xff, 0xFF);


	float lines[18*4] =
	{
	//Origin axis
	0.0f, 0.0f, 0.0f, 1.0f,
	1.0f, 0.0f, 0.0f, 1.0f,

	0.0f, 0.0f, 0.0f, 1.0f,
	0.0f, 1.0f, 0.0f, 1.0f,

	0.0f, 0.0f, 0.0f, 1.0f,
	0.0f, 0.0f, 1.0f, 1.0f,

	//PCA axis
	0.0f, 0.0f, 0.0f, 1.0f,
	c[0], c[1], c[2], 1.0f,

	0.0f, 0.0f, 0.0f, 1.0f,
	c[3], c[4], c[5], 1.0f,

	0.0f, 0.0f, 0.0f, 1.0f,
	c[6], c[7], c[8], 1.0f,

	0.0f, 0.0f, 0.0f, 1.0f,
	0.0f, 0.0f, 0.0f, 1.0f,
	0.0f, 0.0f, 0.0f, 1.0f,
	0.0f, 0.0f, 0.0f, 1.0f,
	0.0f, 0.0f, 0.0f, 1.0f,
	0.0f, 0.0f, 0.0f, 1.0f,
	/*
	0.0f, 0.0f, 0.0f, 1.0f,
	-c[0], -c[1], -c[2], 1.0f,
	0.0f, 0.0f, 0.0f, 1.0f,
	-c[3], -c[4], -c[5], 1.0f,
	0.0f, 0.0f, 0.0f, 1.0f,
	-c[6], -c[7], -c[8], 1.0f,
	*/

	};
	{
		int r;
		r = nng_send (socks[MAIN_NNGSOCK_LINE_POS], lines, 18*4*sizeof(float), 0);
		perror (nng_strerror (r));
	}

	uint32_t line_col[18] =
	{
	//Make origin X axis red:
	RGBA(0xFF, 0x00, 0x00, 0xFF),
	RGBA(0xFF, 0x00, 0x00, 0xFF),
	//Make origin Y axis green:
	RGBA(0x00, 0xFF, 0x00, 0xFF),
	RGBA(0x00, 0xFF, 0x00, 0xFF),
	//Make origin Z axis blue:
	RGBA(0x00, 0x00, 0xFF, 0xFF),
	RGBA(0x00, 0x00, 0xFF, 0xFF),

	//PCA axis
	RGBA(0xFF, 0xFF, 0x00, 0xFF),
	RGBA(0xFF, 0xFF, 0x00, 0xFF),
	RGBA(0xFF, 0xFF, 0x00, 0xFF),
	RGBA(0xFF, 0xFF, 0x00, 0xFF),
	RGBA(0xFF, 0xFF, 0x00, 0xFF),
	RGBA(0xFF, 0xFF, 0x00, 0xFF),
	RGBA(0xFF, 0xFF, 0x00, 0xFF),
	RGBA(0xFF, 0xFF, 0x00, 0xFF),
	RGBA(0xFF, 0xFF, 0x00, 0xFF),
	RGBA(0xFF, 0xFF, 0x00, 0xFF),
	RGBA(0xFF, 0xFF, 0x00, 0xFF),
	RGBA(0xFF, 0xFF, 0x00, 0xFF),

	};
	{
		int r;
		r = nng_send (socks[MAIN_NNGSOCK_LINE_COL], line_col, 18*sizeof(uint32_t), 0);
		perror (nng_strerror (r));
	}




	{
		int r;
		r = nng_send (socks[MAIN_NNGSOCK_POINTCLOUD_POS], point_pos1, LIDAR_WH*4*sizeof(float), 0);
		perror (nng_strerror (r));
		r = nng_send (socks[MAIN_NNGSOCK_POINTCLOUD_COL], pointcol, LIDAR_WH*sizeof(uint32_t), 0);
		perror (nng_strerror (r));
		r = nng_send (socks[MAIN_NNGSOCK_TEX], pix_rgba, IMG_XN*IMG_YN*sizeof(uint32_t), 0);
		perror (nng_strerror (r));
	}

}
