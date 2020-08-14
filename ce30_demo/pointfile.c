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
#include "csc/csc_filecopy.h"

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
	vf32_convolution2d (pix2, pix, xn, yn, kernel, kxn, kyn);
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
	vf32_convolution2d (pix2, pix, xn, yn, kernel, kxn, kyn);
}


float image_best_line_slope (float const p[], uint32_t xn, uint32_t yn, uint32_t yp)
{
	float highscore = 0.0f;
	float k1 = 0.0f;
	float const delta = 0.1f;
	for (float k = -1.0f; k < 1.0f; k += delta)
	{
		float score = 0.0f;
		for (uint32_t y = yp; y < yn-yp; ++y)
		{
			float sum = 0.0f;
			for (uint32_t x = 0; x < xn; ++x)
			{
				//skew in the y-direction by (k) amount:
				float yy = y + x*k;
				if (yy < 0.0f || yy >= yn)
				{
					continue;
				}
				ASSERT (yy >= 0.0f);
				ASSERT (yy < (float)yn);
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


void image_linesum (float p[], uint32_t xn, uint32_t yn, float k, float q[])
{
	for (uint32_t y = 0; y < yn; ++y)
	{
		float sum = 0.0f;
		for (uint32_t x = 0; x < xn; ++x)
		{
			uint32_t yy = (float)y + (float)x*k;
			if (yy < 0){continue;}
			if (yy >= yn){continue;}
			ASSERT (yy >= 0);
			ASSERT (yy < yn);
			uint32_t index = yy*xn + x;
			ASSERT (index < xn*yn);
			sum += p[index];
		}
		//p[y*xn+0] = sum * (1.0f / (float)xn);
		float val = sum;
		q[y] = val;
		//float yy = (float)y + ((float)xn-1.0f)*k;
		//yy = CLAMP (yy, 0, yn);
		//q[(int)yy] = val;
	}
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

		vf32_skip_zero (q, qn, &qi0);
		n = vf32_amount_positive (q, qn, &qi0, 100);
		if (n < 1) {qi++; continue;};

		//skip_zero (q, qn, &qi0);
		n = vf32_amount_negative (q, qn, &qi0, 100);
		if (n < 1 || n > 4) {qi++; continue;};

		//skip_zero (q, qn, &qi0);
		n = vf32_amount_positive (q, qn, &qi0, 100);
		if (n < 1 || n > 4) {qi++; continue;}
		c = qi0 - (n/2) - 1;

		//skip_zero (q, qn, &qi0);
		n = vf32_amount_negative (q, qn, &qi0, 100);
		if (n < 1 || n > 4) {qi++; continue;};

		//skip_zero (q, qn, &qi0);
		n = vf32_amount_positive (q, qn, &qi0, 100);
		if (n < 1) {qi++; continue;};

		qi = qi0 + 10;
		g[gi] = c;
		gi++;
	}
}


void find_pattern2 (float q[], uint32_t qn, uint32_t g[], uint32_t gn, uint32_t r)
{
	for (uint32_t gi = 0; gi < gn; ++gi)
	{
		float qmax = 0.0f;
		uint32_t qimax = 0;
		for (uint32_t qi = 0; qi < qn; ++qi)
		{
			if (qmax < q[qi])
			{
				qmax = q[qi];
				qimax = qi;
			}
		}
		g[gi] = qimax;
		uint32_t qi0 = r <= qimax ? (qimax-r) : 0;
		uint32_t qi1 = MIN (qimax+r, qn);
		for (uint32_t qi = qi0; qi < qi1; ++qi)
		{
			q[qi] = 0.0f;
		}
	}
}



/**
 * @brief Create RGBA image visualisation
 * @param[out] img  RGBA image visual
 * @param[in]  pix  Grayscale image
 * @param[in]  w    Width of the image
 * @param[in]  h    Height of the image
 */
static void image_visual (uint32_t img[], float pix[], uint32_t xn, uint32_t yn, float q[], uint32_t g[], uint32_t m, float k)
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


	for (uint32_t y = 10; y < 11; ++y)
	{
		for (uint32_t x = 0; x < xn; ++x)
		{
			uint32_t yy = (float)y + (float)x*k;
			if (yy < 0){continue;}
			if (yy >= yn){continue;}
			ASSERT (yy >= 0);
			ASSERT (yy < yn);
			uint32_t index = yy*xn + x;
			ASSERT (index < xn*yn);
			img[index] = RGBA (0xF0, 0xF0, 0x00, 0xFF);
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
	vf32_remove_low_values (img2, IMG_XN*IMG_YN);
	image_convolution1 (img3, img2, IMG_XN, IMG_YN);


	float k = image_best_line_slope (img3, IMG_XN, IMG_YN, 10);
	float q[IMG_YN] = {0.0f};
	float q1[IMG_YN] = {0.0f};
	image_linesum (img3, IMG_XN, IMG_YN, k, q);
	vf32_remove_low_values (q, IMG_YN);
	//image_peaks (q, IMG_YN, u);
	uint32_t g[4] = {UINT32_MAX};
	vf32_convolution1d (q, IMG_YN, q1);
	//find_pattern (q, IMG_YN, g, 4);
	find_pattern2 (q1, IMG_YN, g, 2, 16);
	//visual (pix_rgba, pix1, IMG_XN, IMG_YN);
	image_visual (imgv, img1, IMG_XN, IMG_YN, q1, g, 2, k);
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
	ASSERT (argc);
	ASSERT (argv);
	csc_crossos_enable_ansi_color();

	nng_socket socks[MAIN_NNGSOCK_COUNT] = {{0}};
	main_nng_pairdial (socks + MAIN_NNGSOCK_POINTCLOUD_POS, "tcp://localhost:9002");
	main_nng_pairdial (socks + MAIN_NNGSOCK_POINTCLOUD_COL, "tcp://localhost:9003");
	main_nng_pairdial (socks + MAIN_NNGSOCK_TEX,            "tcp://localhost:9004");
	main_nng_pairdial (socks + MAIN_NNGSOCK_VOXEL,          "tcp://localhost:9005");
	main_nng_pairdial (socks + MAIN_NNGSOCK_LINE_POS,       "tcp://localhost:9006");
	main_nng_pairdial (socks + MAIN_NNGSOCK_LINE_COL,       "tcp://localhost:9007");

	chdir ("../ce30_demo/txtpoints2");

	//show ("14_13_57_24145.txt", socks);
	//show ("14_13_55_22538.txt", socks);
	//show ("14_13_53_20565.txt", socks);

#if 1
	FILE * f = popen ("ls", "r");
	ASSERT (f);
	char buf[200] = {'\0'};
	while (1)
	{
		int c;
		do
		{
			c = getchar();
		}
		while (c == '\n');
		switch (c)
		{
		case 'q':
			goto exit_while;
			break;
		case 'n':
			if (fgets (buf, sizeof (buf), f) == NULL) {goto exit_while;};
			buf[strcspn(buf, "\r\n")] = 0;
			printf ("Examining LiDAR point file: %s\n", buf);
			show (buf, socks);
			break;
		case 'c':
			//copy_file (buf, "../txtpoints2");
			break;
		}
	}
exit_while:
	pclose (f);
#endif

	//show ("14_13_57_24254.txt", socks);
	nng_close (socks[MAIN_NNGSOCK_POINTCLOUD_POS]);
	nng_close (socks[MAIN_NNGSOCK_POINTCLOUD_COL]);
	nng_close (socks[MAIN_NNGSOCK_TEX]);
	nng_close (socks[MAIN_NNGSOCK_VOXEL]);
	nng_close (socks[MAIN_NNGSOCK_LINE_POS]);
	nng_close (socks[MAIN_NNGSOCK_LINE_COL]);

	return 0;
}
