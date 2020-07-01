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


#define IMG_XN 20
#define IMG_YN 120


void point_select (uint32_t pointcol[LIDAR_WH], int x, int y, uint32_t color)
{
	int index = LIDAR_INDEX(x,y);
	ASSERT (index < LIDAR_WH);
	printf ("index %i\n", index);
	pointcol[index] = color;
}



void point_pca (float v[], uint32_t v_stride, uint32_t n, float c[3*3], float w[3])
{
	//3 dimensions:
	uint32_t const dim = 3;
	float mean[4] = {0};
	memset (mean, 0, sizeof (float)*dim);
	//Calculate the (mean) coordinate from (v):
	vf32_addv (dim, mean, 0, mean, 0, v, v_stride, n);
	vsf32_mul (dim, mean, mean, 1.0f / (float)n);
	//Move all (v) points to origin using coordinate (mean):
	vf32_subv (dim, v, v_stride, v, v_stride, mean, 0, n);
	//Calculate the covariance matrix (c) from (v):
	memset (c, 0, sizeof (float)*dim*dim);
	m3f32_symmetric_xxt (3, c, 3, v, v_stride, n);
	vsf32_mul (dim*dim, c, c, 1.0f / ((float)n - 1.0f));
	//Calculate the eigen vectors (c) and eigen values (w) from covariance matrix (c):
	LAPACKE_ssyev (LAPACK_COL_MAJOR, 'V', 'U', dim, c, 3, w);
}


/**
 * @brief Filter out points that is not good for finding the ground plane
 * @param[out]    dst           Pointer to the destination array where the elements is to be copied
 * @param[in]     dst_stride    Specifies the byte offset between consecutive elements
 * @param[in]     src           Pointer to the source of data to be copied
 * @param[out]    src_stride    Specifies the byte offset between consecutive elements
 * @param[in,out] n             Is a pointer to an integer related to the number of elements to copy or how many were copied
 * @param[in]     dim           How many dimension in each element
 * @param[in]     k2
 */
void point_filter (float dst[], uint32_t dst_stride, float const src[], uint32_t src_stride, uint32_t *n, uint32_t dim, float k2)
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



void convolution1 (float img[], uint32_t xn, uint32_t yn, uint32_t x0, uint32_t y0, float kernel[], float z)
{
	if (x0 < 1){return;}
	if (y0 < 2){return;}
	if (x0 >= xn-1){return;}
	if (y0 >= yn-2){return;}
	//uint32_t index = ((uint32_t)y0 * xn) + (uint32_t)x0;
	//img[index] += z;
	//return;
	for (int32_t x = -1; x <= 1; ++x)
	{
		for (int32_t y = -2; y <= 2; ++y)
		{
			uint32_t xi = x0 + x;
			uint32_t yi = y0 + y;
			img[yi*xn + xi] += kernel[y*3+x] * z;
			//printf ("(%i %i) : %i\n", x, y, y * xn + x);
		}
	}
	//printf ("\n");
}


void point_project (float pix[], uint32_t xn, uint32_t yn, float v[], uint32_t v_stride, uint32_t x_count)
{
	float counter[IMG_XN*IMG_YN] = {0.0f};
	for (uint32_t i = 0; i < x_count; ++i, v += v_stride)
	{
		float x = v[0]*20.0f + xn/2.0f;
		float y = v[1]*20.0f + yn/2.0f;
		if (x >= xn){continue;}
		if (y >= yn){continue;}
		if (x < 0){continue;}
		if (y < 0){continue;}
		float z = v[2];
		float kernel[]=
		{
		-1.0f, -1.0f, -1.0f,
		 0.0f,  0.0f,  0.0f,
		 2.0f,  2.0f,  2.0f,
		 0.0f,  0.0f,  0.0f,
		-1.0f, -1.0f, -1.0f,
		};
		//z += 10.0f;
		//convolution1 (pix, xn, yn, x, y, kernel, z);
		uint32_t index = ((uint32_t)y * xn) + (uint32_t)x;
		pix[index] += z;
		counter[index] += 1.0f;
		//pix[index] = 0.5f*pix[index] + 0.5f*z;
	}

	for (uint32_t i = 0; i < IMG_XN*IMG_YN; ++i)
	{
		if (counter[i] > 0.0f)
		{
			pix[i] /= counter[i];
		}
		//pix[i] += 10.0f;
	}
}


void image_skitrack_convolution (float pix2[], float const pix[], int32_t xn, int32_t yn)
{
	int32_t kxn = 3;
	int32_t kyn = 5;

	float kernel[3*5] =
	{
	 -1.0f, -4.0f, -1.0f, //Skitrack edge
	  1.0f,  2.0f,  1.0f, //Skitrack dipping
	  1.0f,  2.0f,  1.0f, //Skitrack dipping
	  1.0f,  2.0f,  1.0f, //Skitrack dipping
	 -1.0f, -4.0f, -1.0f, //Skitrack edge
	};
	float kernel1[3*5] =
	{
	 0.0f,  0.0f,  0.0f,
	 0.0f,  0.0f,  0.0f,
	 0.0f,  1.0f,  0.0f,
	 0.0f,  0.0f,  0.0f,
	 0.0f,  0.0f,  0.0f,
	};
	vf32_normalize (kxn*kyn, kernel, kernel);
	for (int32_t y = 1; y < (yn-1); ++y)
	{
		for (int32_t x = 2; x < (xn-2); ++x)
		{
			float sum = 0.0f;
			for (int32_t ky = 0; ky < kyn; ++ky)
			{
				for (int32_t kx = 0; kx < kxn; ++kx)
				{
					int32_t xx = x + kx - 1;
					int32_t yy = y + ky - 2;
					sum += pix[yy * xn + xx] * kernel[ky * kxn + kx];
				}
			}
			pix2[y * xn + x] = sum;
			//pix2[y * xn + x] = pix[y * xn + x];
		}
	}
}



float image_best_line_slope (float const p[], uint32_t xn, uint32_t yn, uint32_t yp)
{
	float highscore = 0.0f;
	float k1 = 0.0f;
	float const delta = 0.1f;
	for (float k = -0.5f; k < 0.5f; k += delta)
	{
		float score = 0.0f;
		for (uint32_t y = yp; y < yn-yp; ++y)
		{
			float sum = 0.0f;
			for (uint32_t x = 0; x < xn; ++x)
			{
				//skew in the y-direction by (k) amount:
				float yy = y + x*k;
				//Sum of noisy pixel will become close to zero:
				//Sum of similiar pixel will become large positive or negative value:
				sum += p[(int)yy*xn + x];
			}
			score += sum*sum;
		}
		if (score > highscore)
		{
			highscore = score;
			k1 = k;
		}
		printf ("sum %f : %f\n", k, score);
	}
	return k1;
}



void image_visual_line (float p[], uint32_t xn, uint32_t yn, uint32_t yp, float k)
{
	for (uint32_t y = yp; y < yn-yp; ++y)
	{
		float sum = 0.0f;
		for (uint32_t x = 0; x < xn; ++x)
		{
			float yy = y + x*k;
			sum += p[(int)yy*xn + x];
		}
		p[y*xn+0] = sum * (1.0f / (float)xn);
	}
}




/**
 * @brief Main algorihtm
 * @param[in,out] x         The pointcloud
 * @param[in]     x_stride  Specifies the byte offset between consecutive point in the pointcloud
 * @param[in]     n         Number of points in the pointcloud
 * @param[out]    pix1      Output image1
 * @param[out]    pix2      Output image2
 * @param[in]     pw        Width of image
 * @param[in]     ph        Height of image
 * @param[out]    c         Eigen vectors
 * @param[out]    w         Eigen values
 */
static void the_algorithm (float x[], uint32_t x_stride, uint32_t n, float pix1[], float pix2[], uint32_t pw, uint32_t ph, float c[], float w[])
{
	//The algorihtm starts here:
	//(Raw points) -> (filter) -> (PCA) -> (Proj2D) -> (2D Convolution)
	point_filter (x, x_stride, x, x_stride, &n, 3, 1.0f);
	point_pca ((float*)x, x_stride, n, c, w);
	printf ("eigen vector:\n"); m3f32_print (c, stdout);
	printf ("eigen value: %f %f %f\n", w[0], w[1], w[2]);
	float rotation[3*3] =
	{
	c[3], c[6], c[0],
	c[4], c[7], c[1],
	c[5], c[8], c[2]
	};
	for (uint32_t i = 0; i < n; ++i)
	{
		float * v = x + (i * x_stride);
		mv3f32_mul (v, rotation, v);
	}
	//cblas_sgemm (CblasColMajor, CblasTrans, CblasNoTrans, 4, point_pos1_count, 4, 1.0f, rotation, 4, point_pos1, 4, 0.0f, point_pos1, 4);
	point_project (pix1, pw, ph, x, x_stride, n);
	image_skitrack_convolution (pix2, pix1, pw, ph);
}




/**
 * @brief Create RGBA image visualisation
 * @param[out] img  The human friendly image
 * @param[in]  pix  The algorihtm friendly image
 * @param[in]  w    Width of the image
 * @param[in]  h    Height of the image
 */
static void visual (uint32_t img[], float pix[], uint32_t w, uint32_t h)
{
	for (uint32_t i = 0; i < w*h; ++i)
	{
		uint8_t r = CLAMP ((-pix[i])*3000.0f, 0.0f, 255.0f);
		uint8_t g = CLAMP ((pix[i])*3000.0f, 0.0f, 255.0f);
		//uint8_t r = CLAMP (pix1[i]*1000.0f, 0.0f, 255.0f);
		//uint8_t g = CLAMP (-pix1[i]*1000.0f, 0.0f, 255.0f);
		img[i] = RGBA (r, g, 0x00, 0x44);
		//pix_rgba[i] = RGBA (pix1[i] > 0.4f ? 0xFF : 0x00, 0x00, 0x00, 0xFF);
	}
}



int main()
{
	csc_crossos_enable_ansi_color();
	char const * txtpoint = csc_malloc_file ("../ce30_demo/txtpoints/14_14_02_29138.txt");

	float point_pos1[LIDAR_WH*POINT_STRIDE] = {0.0f};
	uint32_t point_pos1_count = LIDAR_WH;

	uint32_t pointcol[LIDAR_WH] = {0};
	for (int i = 0; i < LIDAR_WH; ++i) {pointcol[i] = RGBA (0xFF, 0xFF, 0xFF, 0xFF);}

	points_read (txtpoint, point_pos1, &point_pos1_count);
	//points_print (point, n);

	nng_socket socks[MAIN_NNGSOCK_COUNT] = {{0}};
	main_nng_pairdial (socks + MAIN_NNGSOCK_POINTCLOUD_POS, "tcp://192.168.1.176:9002");
	main_nng_pairdial (socks + MAIN_NNGSOCK_POINTCLOUD_COL, "tcp://192.168.1.176:9003");
	main_nng_pairdial (socks + MAIN_NNGSOCK_TEX,            "tcp://192.168.1.176:9004");
	main_nng_pairdial (socks + MAIN_NNGSOCK_VOXEL,          "tcp://192.168.1.176:9005");
	main_nng_pairdial (socks + MAIN_NNGSOCK_LINE_POS,       "tcp://192.168.1.176:9006");
	main_nng_pairdial (socks + MAIN_NNGSOCK_LINE_COL,       "tcp://192.168.1.176:9007");

	/*
	for (uint32_t i = 0; i < point_pos1_count; ++i)
	{
		printf ("%f %f %f\n", point_pos1[i*POINTS_DIM + 0], point_pos1[i*POINTS_DIM + 1], point_pos1[i*POINTS_DIM + 2]);
	}
	*/



	float pix1[IMG_XN*IMG_YN] = {0};
	float pix2[IMG_XN*IMG_YN] = {0};
	uint32_t pix_rgba[IMG_XN*IMG_YN] = {0};
	float c[3*3];
	float w[3];
	the_algorithm (point_pos1, POINT_STRIDE, point_pos1_count, pix1, pix2, IMG_XN, IMG_YN, c, w);
	float k = image_best_line_slope (pix2, IMG_XN, IMG_YN, 10);
	image_visual_line (pix2, IMG_XN, IMG_YN, 10, k);
	//visual (pix_rgba, pix1, IMG_XN, IMG_YN);
	visual (pix_rgba, pix2, IMG_XN, IMG_YN);

	//pix_rgba[105*IMG_XN + 12] |= RGBA(0x00, 0x66, 0x00, 0x00);
	//pix_rgba[0*IMG_XN + 1] |= RGBA(0x00, 0xFF, 0x00, 0xFF);
	//pix_rgba[2*IMG_XN + 0] |= RGBA(0x00, 0xFF, 0xff, 0xFF);
	//pix_rgba[2*IMG_XN + 1] |= RGBA(0x00, 0xFF, 0xff, 0xFF);



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
	//Make origin Y axis green:
	//Make origin Z axis blue:
	RGBA(0xFF, 0x00, 0x00, 0xFF),
	RGBA(0xFF, 0x00, 0x00, 0xFF),
	RGBA(0x00, 0xFF, 0x00, 0xFF),
	RGBA(0x00, 0xFF, 0x00, 0xFF),
	RGBA(0x00, 0x00, 0xFF, 0xFF),
	RGBA(0x00, 0x00, 0xFF, 0xFF),

	//PCA axis
	RGBA(0xFF, 0xFF, 0xFF, 0x33),
	RGBA(0xFF, 0xFF, 0xFF, 0x33),
	RGBA(0xFF, 0xFF, 0xFF, 0x33),
	RGBA(0xFF, 0xFF, 0xFF, 0x33),
	RGBA(0xFF, 0xFF, 0xFF, 0x33),
	RGBA(0xFF, 0xFF, 0xFF, 0x33),
	RGBA(0xFF, 0xFF, 0xFF, 0x33),
	RGBA(0xFF, 0xFF, 0xFF, 0x33),
	RGBA(0xFF, 0xFF, 0xFF, 0x33),
	RGBA(0xFF, 0xFF, 0xFF, 0x33),
	RGBA(0xFF, 0xFF, 0xFF, 0x33),
	RGBA(0xFF, 0xFF, 0xFF, 0x33),

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
