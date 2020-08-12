#include <unistd.h>
#include <stdio.h>

//git clone https://github.com/nanomsg/nng
//cd nng && mkdir build && cd build
//cmake -G"MSYS Makefiles" .. -DCMAKE_INSTALL_PREFIX="C:\msys64\mingw64"
//pacman -R cmake
//pacman -S mingw-w64-x86_64-cmake
//mingw32-make -j4
//mingw32-make test
//mingw32-make install
//-lnng
#include <nng/nng.h>
#include <nng/protocol/pair0/pair.h>
#include <nng/supplemental/util/platform.h>

//pacman -S mingw64/mingw-w64-x86_64-openblas
//-lopenblas
#include <OpenBLAS/lapack.h>
#include <OpenBLAS/lapacke.h>
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


void point_mean (float x[], uint32_t ldx, uint32_t n, float mean[3])
{
	//3 dimensions:
	uint32_t const dim = 3;
	memset (mean, 0, sizeof (float)*dim);
	//Calculate the (mean) coordinate from (v):
	vf32_addv (dim, mean, 0, mean, 0, x, ldx, n);
	vsf32_mul (dim, mean, mean, 1.0f / (float)n);
	//Move all (v) points to origin using coordinate (mean):
	vf32_subv (dim, x, ldx, x, ldx, mean, 0, n);
}


void point_covariance (float v[], uint32_t v_stride, uint32_t n, float c[3*3])
{
	//3 dimensions:
	uint32_t const dim = 3;
	//Calculate the covariance matrix (c) from (v):
	memset (c, 0, sizeof (float)*dim*dim);
	m3f32_symmetric_xxt (3, c, 3, v, v_stride, n);
	vsf32_mul (dim*dim, c, c, 1.0f / ((float)n - 1.0f));
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


void point_project (float pix[], float imgf[], uint32_t xn, uint32_t yn, float v[], uint32_t v_stride, uint32_t x_count)
{
	for (uint32_t i = 0; i < x_count; ++i, v += v_stride)
	{
		//v[2] += 1.0f;
		//(x,y) becomes the pixel position:
		//Set origin in the middle of the image and 20 pixels becomes 1 meter:
		float x = v[0]*20.0f + xn/2.0f;
		float y = v[1]*20.0f + yn/2.0f;
		//z-value becomes the pixel value:
		float z = v[2];
		//Crop the pointcloud to the size of the image;
		if (x >= xn){continue;}
		if (y >= yn){continue;}
		if (x < 0){continue;}
		if (y < 0){continue;}
		//Convert (x,y) to index row-major:
		uint32_t index = ((uint32_t)y * xn) + (uint32_t)x;
		//z += 10.0f;
		//If multiple points land on one pixel then it will be accumalted but it will also be normalized later on:
		pix[index] += z;
		imgf[index] += 1.0f;
		//pix[index] = 0.5f*pix[index] + 0.5f*z;
	}

	//Normalize every non zero pixel:
	for (uint32_t i = 0; i < IMG_XN*IMG_YN; ++i)
	{
		if (imgf[i] > 0.0f)
		{
			pix[i] /= imgf[i];
		}
	}

	for (uint32_t i = 0; i < IMG_XN*IMG_YN; ++i)
	{
		//Gradient convolution could be applied later so this statement will have no effect:
		//It is important that this statement does not affect the end result:
		//This statement test scenories where average pointcloud z-position is far of origin:
		//pix[i] += 10.0f;
	}
}


void image_convolution (float pix2[], float const pix[], int32_t xn, int32_t yn, float k[], int32_t kxn, int32_t kyn)
{
	int32_t kxn0 = kxn / 2;
	int32_t kyn0 = kyn / 2;
	//printf ("kxn0 %i\n", kxn0);
	//printf ("kyn0 %i\n", kyn0);
	for (int32_t y = kyn0; y < (yn-kyn0); ++y)
	{
		for (int32_t x = kxn0; x < (xn-kxn0); ++x)
		{
			float sum = 0.0f;
			for (int32_t ky = 0; ky < kyn; ++ky)
			{
				for (int32_t kx = 0; kx < kxn; ++kx)
				{
					int32_t xx = x + kx - kxn0;
					int32_t yy = y + ky - kyn0;
					sum += pix[yy * xn + xx] * k[ky * kxn + kx];
				}
			}
			pix2[y * xn + x] = sum;
			//pix2[y * xn + x] = pix[y * xn + x];
		}
	}
}



void image_skitrack_convolution (float pix2[], float const pix[], int32_t xn, int32_t yn)
{
	///*
	int32_t kxn = 3;
	int32_t kyn = 5;

	float kernel[3*5] =
	{
	 -2.0f, -4.0f, -2.0f, //Skitrack edge
	  1.0f,  2.0f,  1.0f, //Skitrack dipping
	  2.0f,  3.0f,  2.0f, //Skitrack dipping
	  1.0f,  2.0f,  1.0f, //Skitrack dipping
	 -2.0f, -4.0f, -2.0f, //Skitrack edge
	};
	//*/
	/*
	float kernel1[3*5] =
	{
	 0.0f,  0.0f,  0.0f,
	 0.0f,  0.0f,  0.0f,
	 0.0f,  1.0f,  0.0f,
	 0.0f,  0.0f,  0.0f,
	 0.0f,  0.0f,  0.0f,
	};
	*/
	/*
	int32_t kxn = 1;
	int32_t kyn = 11;
	float kernel[11] =
	{
	 1.0f,
	 1.0f,
	-1.0f,
	-2.0f,
	-1.0f,
	 2.0f,
	 2.0f,
	-1.0f,
	-2.0f,
	-1.0f,
	 1.0f,
	};
	*/
	vf32_normalize (kxn*kyn, kernel, kernel);
	image_convolution (pix2, pix, xn, yn, kernel, kxn, kyn);
}


void image_convolution1 (float pix2[], float const pix[], int32_t xn, int32_t yn)
{
	int32_t kxn = 3;
	int32_t kyn = 3;

	float kernel[3*3] =
	{
	 1.0f, 1.0f, 1.0f,
	 1.0f, 2.0f, 1.0f,
	 1.0f, 1.0f, 1.0f,
	};
	vf32_normalize (kxn*kyn, kernel, kernel);
	image_convolution (pix2, pix, xn, yn, kernel, kxn, kyn);
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
		printf ("sum:  %+f : %f\n", k, score);
	}
	printf ("best: %+f : %f\n", k1, highscore);
	return k1;
}


/**
 * @brief
 * @param p
 * @param xn
 * @param yn
 * @param yp
 * @param k
 */
void image_visual_line (float p[], uint32_t xn, uint32_t yn, uint32_t yp, float k, float q[])
{
	for (uint32_t y = yp; y < yn-yp; ++y)
	{
		float sum = 0.0f;
		for (uint32_t x = 0; x < xn; ++x)
		{
			float yy = y + x*k;
			sum += p[(int)yy*xn + x];
		}
		//p[y*xn+0] = sum * (1.0f / (float)xn);
		float val = sum;
		q[y] = val;
		float yy = (float)y + ((float)xn-1.0f)*k;
		yy = CLAMP (yy, 0, yn);
		//q[(int)yy] = val;
	}
}


void image_peaks (float const q[], uint32_t n, float u[])
{
	uint32_t kn = 13;
	uint32_t kn0 = kn / 2;
	float k[13] = {1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 2.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f};
	vf32_normalize (kn, k, k);

	for (uint32_t i = kn0; i < n-kn0; ++i)
	{
		float sum = 0.0f;
		for (uint32_t j = 0; j < kn; ++j)
		{
			sum += q[i - kn0 + j] * k[j];
		}
		u[i] = sum;
	}
}


uint32_t search_positive (float q[], uint32_t qn, uint32_t * qi, uint32_t count)
{
	uint32_t n = 0;
	while (count--)
	{
		if ((*qi) >= qn){break;}
		if (q[(*qi)] <= 0.0f){break;}
		(*qi)++;
		n++;
	}
	return n;
}

uint32_t search_negative (float q[], uint32_t qn, uint32_t * qi, uint32_t count)
{
	uint32_t n = 0;
	while (count--)
	{
		if ((*qi) >= qn){break;}
		if (q[(*qi)] >= 0.0f){break;}
		(*qi)++;
		n++;
	}
	return n;
}


void filter (float q[], uint32_t qn)
{
	float pos = 0.0f;
	float neg = 0.0f;
	float pos_n = 0;
	float neg_n = 0;
	for (uint32_t i = 0; i < qn; ++i)
	{
		if (q[i] > 0.0f)
		{
			pos += q[i];
			pos_n += 1.0f;
		}
		else if (q[i] < 0.0f)
		{
			neg += q[i];
			neg_n += 1.0f;
		}
	}
	pos /= pos_n;
	neg /= neg_n;
	for (uint32_t i = 0; i < qn; ++i)
	{
		if ((q[i] > 0.0f) && (q[i] < pos))
		{
			q[i] = 0.0f;
		}
		if ((q[i] < 0.0f) && (q[i] > neg))
		{
			q[i] = 0.0f;
		}
	}
}



uint32_t skip_zero (float q[], uint32_t qn, uint32_t * qi)
{
	uint32_t n = 0;
	while (1)
	{
		if ((*qi) >= qn){break;}
		if (q[(*qi)] != 0.0f){break;}
		(*qi)++;
		n++;
	}
	return n;
}


void find_pattern (float q[], uint32_t qn, uint32_t g[], uint32_t gn)
{
	uint32_t gi = 0;
	uint32_t qi = 0;
	uint32_t n;
	uint32_t c = 0;
	while (1)
	{
		if (gi >= gn) {break;};
		if (qi >= qn) {break;};

		uint32_t qi0 = qi;

		skip_zero (q, qn, &qi0);
		n = search_positive (q, qn, &qi0, 100);
		if (n < 1) {qi++; continue;};

		skip_zero (q, qn, &qi0);
		n = search_negative (q, qn, &qi0, 100);
		if (n < 1 || n > 4) {qi++; continue;};

		skip_zero (q, qn, &qi0);
		n = search_positive (q, qn, &qi0, 100);
		if (n < 1 || n > 4) {qi++; continue;}
		c = qi0 - (n/2);

		skip_zero (q, qn, &qi0);
		n = search_negative (q, qn, &qi0, 100);
		if (n < 1 || n > 4) {qi++; continue;};

		skip_zero (q, qn, &qi0);
		n = search_positive (q, qn, &qi0, 100);
		if (n < 1) {qi++; continue;};

		qi = qi0 + 10;
		g[gi] = c;
		gi++;
	}
}



void find_pattern2 (float q[], uint32_t qn, uint32_t g[], uint32_t gn, uint32_t r)
{
	float p[15] = {1, 1, 0,-1,-1,-1, 1, 2, 1,-1,-1,-1, 0, 1, 1};
	r=15;
	for (uint32_t gi = 0; gi < gn; ++gi)
	{
		float max = 0.0f;
		uint32_t imax = 0;
		for (uint32_t i = (r/2); i < qn-(r/2); ++i)
		{
			float sum = 0.0f;
			for (uint32_t j = 0; j < r; ++j)
			{
				sum += q[i]*p[j];
			}
			if (max < sum)
			{
				max = sum;
				imax = i;
				q[imax-1] = 0.0f;
				q[imax] = 0.0f;
				q[imax+1] = 0.0f;
			}
		}
		g[gi] = imax;
	}
}


/**
 * @brief Create RGBA image visualisation
 * @param[out] img  RGBA image visual
 * @param[in]  pix  Grayscale image
 * @param[in]  w    Width of the image
 * @param[in]  h    Height of the image
 */
static void image_visual (uint32_t img[], float pix[], uint32_t xn, uint32_t yn, float q[], uint32_t g[], uint32_t m)
{
	for (uint32_t i = 0; i < xn*yn; ++i)
	{
		//Negatives becomes red and positives becomes greeen:
		uint8_t r = CLAMP ((-pix[i])*3000.0f, 0.0f, 255.0f);
		uint8_t g = CLAMP ((pix[i])*3000.0f, 0.0f, 255.0f);
		//uint8_t r = CLAMP (pix1[i]*1000.0f, 0.0f, 255.0f);
		//uint8_t g = CLAMP (-pix1[i]*1000.0f, 0.0f, 255.0f);
		img[i] = RGBA (r, g, 0x00, 0xFF);
		//pix_rgba[i] = RGBA (pix1[i] > 0.4f ? 0xFF : 0x00, 0x00, 0x00, 0xFF);
	}
	for (uint32_t y = 0; y < yn; ++y)
	{
		uint8_t r = CLAMP ((-q[y])*300.0f, 0.0f, 255.0f);
		uint8_t g = CLAMP ((q[y])*300.0f, 0.0f, 255.0f);
		img[y*xn+0] = RGBA (r, g, 0x00, 0xFF);
		//img[y*xn+xn-1] = RGBA (r, g, 0x00, 0xFF);
	}

	for (uint32_t i = 0; i < m; ++i)
	{
		if (g[i] < yn)
		{
			img[g[i]*xn+1] = RGBA (0x00, 0x00, 0xFF, 0xFF);
		}
	}
}


void show (const char * filename, nng_socket socks[])
{
	char const * txtpoint = csc_malloc_file (filename);
	float point_pos1[LIDAR_WH*POINT_STRIDE] = {0.0f};
	uint32_t point_pos1_count = LIDAR_WH;
	points_read (txtpoint, point_pos1, &point_pos1_count);
	free ((void*)txtpoint);
	//points_print (point, n);

	uint32_t pointcol[LIDAR_WH] = {0};
	for (int i = 0; i < LIDAR_WH; ++i) {pointcol[i] = RGBA (0xFF, 0xFF, 0xFF, 0xFF);}

	/*
	for (uint32_t i = 0; i < point_pos1_count; ++i)
	{
		printf ("%f %f %f\n", point_pos1[i*POINTS_DIM + 0], point_pos1[i*POINTS_DIM + 1], point_pos1[i*POINTS_DIM + 2]);
	}
	*/

	float img1[IMG_XN*IMG_YN] = {0.0f};//Projected points
	float img2[IMG_XN*IMG_YN] = {0.0f};//Convolution from img1
	float img3[IMG_XN*IMG_YN] = {0.0f};//Convolution from img2
	float imgf[IMG_XN*IMG_YN] = {0.0f};//Used for normalizing pixel
	uint32_t imgv[IMG_XN*IMG_YN] = {0};//Used for visual confirmation that the algorithm works
	float c[3*3];//Covariance matrix first then 3x eigen vectors
	float w[3];//Eigen values
	float pc_mean[3];

	//The algorihtm starts here:
	//(Raw points) -> (filter) -> (PCA) -> (Proj2D) -> (2D Convolution)
	point_filter (point_pos1, POINT_STRIDE, point_pos1, POINT_STRIDE, &point_pos1_count, 3, 1.0f);
	point_mean (point_pos1, POINT_STRIDE, point_pos1_count, pc_mean);
	point_covariance ((float*)point_pos1, POINT_STRIDE, point_pos1_count, c);
	//Calculate the eigen vectors (c) and eigen values (w) from covariance matrix (c):
	//https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dsyev.htm
	LAPACKE_ssyev (LAPACK_COL_MAJOR, 'V', 'U', 3, c, 3, w);
	//LAPACK_ssyev ();
	printf ("eigen vector:\n"); m3f32_print (c, stdout);
	printf ("eigen value: %f %f %f\n", w[0], w[1], w[2]);
	float rotation[3*3] =
	{
	c[3], c[6], c[0],
	c[4], c[7], c[1],
	c[5], c[8], c[2]
	};
	//TODO: Do a matrix matrix multiplication instead of matrix vector multiplication:
	for (uint32_t i = 0; i < point_pos1_count; ++i)
	{
		float * v = point_pos1 + (i * POINT_STRIDE);
		mv3f32_mul (v, rotation, v);
	}
	//cblas_sgemm (CblasColMajor, CblasTrans, CblasNoTrans, 4, point_pos1_count, 4, 1.0f, rotation, 4, point_pos1, 4, 0.0f, point_pos1, 4);
	point_project (img1, imgf, IMG_XN, IMG_YN, point_pos1, POINT_STRIDE, point_pos1_count);
	image_skitrack_convolution (img2, img1, IMG_XN, IMG_YN);
	filter (img2, IMG_XN*IMG_YN);
	image_convolution1 (img3, img2, IMG_XN, IMG_YN);
	//filter (img3, IMG_XN*IMG_YN);


	float k = image_best_line_slope (img3, IMG_XN, IMG_YN, 10);
	float q[IMG_YN] = {0.0f};
	image_visual_line (img3, IMG_XN, IMG_YN, 10, k, q);
	filter (q, IMG_YN);
	//image_peaks (q, IMG_YN, u);
	uint32_t g[4] = {UINT32_MAX};
	find_pattern (q, IMG_YN, g, 4);
	//visual (pix_rgba, pix1, IMG_XN, IMG_YN);
	image_visual (imgv, img1, IMG_XN, IMG_YN, q, g, 4);
	//pix_rgba[105*IMG_XN + 12] |= RGBA(0x00, 0x66, 0x00, 0x00);
	//pix_rgba[0*IMG_XN + 1] |= RGBA(0x00, 0xFF, 0x00, 0xFF);
	//pix_rgba[2*IMG_XN + 0] |= RGBA(0x00, 0xFF, 0xff, 0xFF);
	//pix_rgba[2*IMG_XN + 1] |= RGBA(0x00, 0xFF, 0xff, 0xFF);



	float lines[18*4] =
	{
	//Origin axis
	0.0f, 0.0f, 0.0f, 1.0f, //Origin axis 1 start
	1.0f, 0.0f, 0.0f, 1.0f, //Origin axis 1 end
	0.0f, 0.0f, 0.0f, 1.0f, //Origin axis 2 start
	0.0f, 1.0f, 0.0f, 1.0f, //Origin axis 2 end
	0.0f, 0.0f, 0.0f, 1.0f, //Origin axis 3 start
	0.0f, 0.0f, 1.0f, 1.0f, //Origin axis 3 end

	//PCA axis
	0.0f, 0.0f, 0.0f, 1.0f, //PCA axis 1 start
	c[0], c[1], c[2], 1.0f, //PCA axis 1 end
	0.0f, 0.0f, 0.0f, 1.0f, //PCA axis 2 start
	c[3], c[4], c[5], 1.0f, //PCA axis 2 end
	0.0f, 0.0f, 0.0f, 1.0f, //PCA axis 3 start
	c[6], c[7], c[8], 1.0f, //PCA axis 3 end

	//TODO: What is this?
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


	uint32_t line_col[18] =
	{
	//Origin axis:
	RGBA(0xFF, 0x00, 0x00, 0xFF), //Origin axis 1 start
	RGBA(0xFF, 0x00, 0x00, 0xFF), //Origin axis 1 end
	RGBA(0x00, 0xFF, 0x00, 0xFF), //Origin axis 2 start
	RGBA(0x00, 0xFF, 0x00, 0xFF), //Origin axis 2 end
	RGBA(0x00, 0x00, 0xFF, 0xFF), //Origin axis 3 start
	RGBA(0x00, 0x00, 0xFF, 0xFF), //Origin axis 3 end

	//PCA axis colors:
	RGBA(0xFF, 0xFF, 0xFF, 0x33), //PCA axis 1 start
	RGBA(0xFF, 0xFF, 0xFF, 0x33), //PCA axis 1 end
	RGBA(0xFF, 0xFF, 0xFF, 0x33), //PCA axis 2 start
	RGBA(0xFF, 0xFF, 0xFF, 0x33), //PCA axis 2 end
	RGBA(0xFF, 0xFF, 0xFF, 0x33), //PCA axis 3 start
	RGBA(0xFF, 0xFF, 0xFF, 0x33), //PCA axis 3 end

	//TODO: What is this?
	RGBA(0xFF, 0xFF, 0xFF, 0x33),
	RGBA(0xFF, 0xFF, 0xFF, 0x33),
	RGBA(0xFF, 0xFF, 0xFF, 0x33),
	RGBA(0xFF, 0xFF, 0xFF, 0x33),
	RGBA(0xFF, 0xFF, 0xFF, 0x33),
	RGBA(0xFF, 0xFF, 0xFF, 0x33),

	};


	//Send visual information to the graphic server:
	{
		int r;
		r = nng_send (socks[MAIN_NNGSOCK_LINE_POS], lines, 18*4*sizeof(float), 0);
		perror (nng_strerror (r));
		r = nng_send (socks[MAIN_NNGSOCK_LINE_COL], line_col, 18*sizeof(uint32_t), 0);
		perror (nng_strerror (r));
		r = nng_send (socks[MAIN_NNGSOCK_POINTCLOUD_POS], point_pos1, LIDAR_WH*4*sizeof(float), 0);
		perror (nng_strerror (r));
		r = nng_send (socks[MAIN_NNGSOCK_POINTCLOUD_COL], pointcol, LIDAR_WH*sizeof(uint32_t), 0);
		perror (nng_strerror (r));
		r = nng_send (socks[MAIN_NNGSOCK_TEX], imgv, IMG_XN*IMG_YN*sizeof(uint32_t), 0);
		perror (nng_strerror (r));
	}

}




int main (int argc, char const * argv[])
{
	csc_crossos_enable_ansi_color();

	nng_socket socks[MAIN_NNGSOCK_COUNT] = {{0}};
	main_nng_pairdial (socks + MAIN_NNGSOCK_POINTCLOUD_POS, "tcp://localhost:9002");
	main_nng_pairdial (socks + MAIN_NNGSOCK_POINTCLOUD_COL, "tcp://localhost:9003");
	main_nng_pairdial (socks + MAIN_NNGSOCK_TEX,            "tcp://localhost:9004");
	main_nng_pairdial (socks + MAIN_NNGSOCK_VOXEL,          "tcp://localhost:9005");
	main_nng_pairdial (socks + MAIN_NNGSOCK_LINE_POS,       "tcp://localhost:9006");
	main_nng_pairdial (socks + MAIN_NNGSOCK_LINE_COL,       "tcp://localhost:9007");

	chdir ("../ce30_demo/txtpoints");
	///*
	FILE * f = popen ("ls", "r");
	ASSERT (f);
	char buf[200] = {'\0'};
	while (fgets (buf, sizeof (buf), f) != NULL)
	{
		buf[strcspn(buf, "\r\n")] = 0;
		printf("OUTPUT: (%s)\n", buf);
		show (buf, socks);
		sleep (1);
	}
	pclose (f);
	//*/
	//show ("14_13_57_24254.txt", socks);
	nng_close (socks[MAIN_NNGSOCK_POINTCLOUD_POS]);
	nng_close (socks[MAIN_NNGSOCK_POINTCLOUD_COL]);
	nng_close (socks[MAIN_NNGSOCK_TEX]);
	nng_close (socks[MAIN_NNGSOCK_VOXEL]);
	nng_close (socks[MAIN_NNGSOCK_LINE_POS]);
	nng_close (socks[MAIN_NNGSOCK_LINE_COL]);

	return 0;
}
