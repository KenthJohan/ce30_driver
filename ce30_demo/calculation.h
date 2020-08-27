/*
In 2D computer graphics, a pixel represents a value on a regular grid in two-dimensional space.
In 3D computer graphics, a voxel represents a value on a regular grid in three-dimensional space.
*/

#pragma once

#include <nng/nng.h>
#include <nng/protocol/pair0/pair.h>
#include <nng/supplemental/util/platform.h>

#include <stdio.h>

//pacman -S mingw64/mingw-w64-x86_64-openblas
//-lopenblas
#include <OpenBLAS/lapack.h>
#include <OpenBLAS/lapacke.h>
#include <OpenBLAS/cblas.h>

#include "csc/csc_debug_nng.h"
#include "csc/csc_math.h"
#include "csc/csc_linmat.h"
#include "csc/csc_m3f32.h"
#include "csc/csc_crossos.h"
#include "csc/csc_malloc_file.h"

#define RGBA(r,g,b,a) (((r) << 0) | ((g) << 8) | ((b) << 16) | ((a) << 24))

#define POINT_STRIDE 4

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
	r = nng_dial (*sock, address, NULL, NNG_FLAG_NONBLOCK);
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


void points_read_filename (char const * filename, float p[], uint32_t *n)
{
	ASSERT_PARAM_NOTNULL (filename);
	ASSERT_PARAM_NOTNULL (p);
	ASSERT_PARAM_NOTNULL (n);
	char const * text = csc_malloc_file (filename);
	points_read (text, p, n);
	free ((void*)text);
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





#define IMG_XN 20
#define IMG_YN 120


void vf32_project_2d_to_1d (float p[], uint32_t xn, uint32_t yn, float k, float q[])
{
	for (uint32_t y = 0; y < yn; ++y)
	{
		float sum = 0.0f;
		for (uint32_t x = 0; x < xn; ++x)
		{
			float yy = (float)y + (float)x*k;
			if (yy < 0.0f){continue;}
			if (yy >= (float)yn){continue;}
			ASSERT (yy >= 0.0f);
			ASSERT (yy < (float)yn);
			uint32_t index = (uint32_t)yy*xn + x;
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


void vf32_project_2d_to_1d_pn (float const p[], uint32_t xn, uint32_t yn, float k, float q[])
{
	for (uint32_t y = 0; y < yn; ++y)
	{
		float sump = 0.0f;
		float sumn = 0.0f;
		for (uint32_t x = 0; x < xn; ++x)
		{
			float yy = (float)y + (float)x*k;
			if (yy < 0.0f){continue;}
			if (yy >= (float)yn){continue;}
			ASSERT (yy >= 0.0f);
			ASSERT (yy < (float)yn);
			uint32_t index = (uint32_t)yy*xn + x;
			ASSERT (index < xn*yn);
			if (p[index] > 0.0f)
			{
				sump += 1;
			}
			if (p[index] < 0.0f)
			{
				sumn += 1;
			}
		}
		//p[y*xn+0] = sum * (1.0f / (float)xn);
		float val = (sump - sumn) / xn;
		q[y] = val;
		//float yy = (float)y + ((float)xn-1.0f)*k;
		//yy = CLAMP (yy, 0, yn);
		//q[(int)yy] = val;
	}
}



float vf32_most_common_line (float const p[], uint32_t xn, uint32_t yn, uint32_t yp)
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
				uint32_t index = (uint32_t)yy*xn + x;
				ASSERT (index < xn*yn);
				//Sum of noisy pixel will become close to zero:
				//Sum of similiar pixel will become large positive or negative value:
				//TODO: Do not count undefined pixels:
				sum += p[index];
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



float vf32_most_common_line2 (float const p[], uint32_t xn, uint32_t yn, float q[])
{
	float max = 0.0f;
	float k1 = 0.0f;
	float const delta = 0.1f;
	for (float k = -1.0f; k < 1.0f; k += delta)
	{
		vf32_project_2d_to_1d_pn (p, xn, yn, k, q);
		float sum = 0.0f;
		for (uint32_t i = 0; i < yn; ++i)
		{
			sum += q[i]*q[i];
		}
		if (sum > max)
		{
			max = sum;
			k1 = k;
		}
	}
	printf ("max: %+f : %f\n", k1, max);
	return k1;
}








void point_select (uint32_t pointcol[LIDAR_WH], int x, int y, uint32_t color)
{
	int index = LIDAR_INDEX(x,y);
	ASSERT (index < LIDAR_WH);
	printf ("index %i\n", index);
	pointcol[index] = color;
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


void point_to_pixel (float const p[4], uint32_t xn, uint32_t yn, float * pixel, float * x, float * y)
{
	float const sx = 20.0f;
	float const sy = 20.0f;
	(*x) = p[0]*sx + xn/2.0f;
	(*y) = p[1]*sy + yn/2.0f;
	//z-value becomes the pixel value:
	(*pixel) = p[2];
}


void pixel_to_point (float p[4], uint32_t xn, uint32_t yn, float pixel, float x, float y)
{
	float const sx = 20.0f;
	float const sy = 20.0f;
	//x = p*sx + xn/2
	//x - xn/2 = p*sx
	//(x - xn/2)/sx = p
	p[0] = (x - xn/2) / sx;
	p[1] = (y - yn/2) / sy;
	p[2] = pixel;
}




void point_project (float pix[], float imgf[], uint32_t xn, uint32_t yn, float v[], uint32_t v_stride, uint32_t x_count)
{
	for (uint32_t i = 0; i < x_count; ++i, v += v_stride)
	{
		//v[2] += 1.0f;
		//(x,y) becomes the pixel position:
		//Set origin in the middle of the image and 20 pixels becomes 1 meter:
		//z-value becomes the pixel value:
		float x;
		float y;
		float z;
		point_to_pixel (v, xn, yn, &z, &x, &y);
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



/**
 * @brief Create RGBA image visualisation
 * @param[out] img  RGBA image visual
 * @param[in]  pix  Grayscale image
 * @param[in]  w    Width of the image
 * @param[in]  h    Height of the image
 */
static void image_visual (uint32_t img[], float pix[], uint32_t xn, uint32_t yn, float q1[], float q2[], uint32_t g[], uint32_t m, float k)
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
		if (q1[y])
		{
			uint8_t r = CLAMP ((-q1[y])*100.0f, 0.0f, 255.0f);
			uint8_t g = CLAMP ((q1[y])*100.0f, 0.0f, 255.0f);
			img[y*xn+1] = RGBA (r, g, 0x00, 0xFF);
		}
		else
		{
			img[y*xn+1] = RGBA (0x22, 0x22, 0x22, 0xFF);
		}
		//img[y*xn+xn-1] = RGBA (r, g, 0x00, 0xFF);
	}

	for (uint32_t y = 0; y < yn; ++y)
	{
		if (q2[y])
		{
			uint8_t r = CLAMP ((-q2[y])*100.0f, 0.0f, 255.0f);
			uint8_t g = CLAMP ((q2[y])*100.0f, 0.0f, 255.0f);
			img[y*xn+0] = RGBA (r, g, 0x00, 0xFF);
		}
		else
		{
			img[y*xn+0] = RGBA (0x22, 0x22, 0x22, 0xFF);
		}
		//img[y*xn+xn-1] = RGBA (r, g, 0x00, 0xFF);
	}



	for (uint32_t i = 0; i < m; ++i)
	{
		if (g[i] < yn)
		{
			uint32_t y = g[i];
			for (uint32_t x = 0; x < xn; ++x)
			{
				float yy = (float)y + (float)x*k;
				if (yy < 0.0f){continue;}
				if (yy >= (float)yn){continue;}
				ASSERT (yy >= 0);
				ASSERT (yy < (float)yn);
				uint32_t index = (uint32_t)yy * xn + x;
				ASSERT (index < xn*yn);
				img[index] |= RGBA (0x00, 0x00, 0x66, 0xFF);
			}
		}
	}
}



lapack_int m3f32_lapacke_inverse (float *A, unsigned n)
{
	int ipiv[3*3];
	lapack_int ret;
	ret =  LAPACKE_sgetrf (LAPACK_COL_MAJOR, n, n, A, n, ipiv);
	if (ret !=0)
	{
		return ret;
	}
	ret = LAPACKE_sgetri (LAPACK_COL_MAJOR, n, A, n, ipiv);
	return ret;
}







struct skitrack1
{
	uint32_t pc_count;
	float pc[LIDAR_WH*POINT_STRIDE];
	float w[3];//Eigen values
	float c[3*3];//Covariance matrix first then 3x eigen vectors
	float crot[3*3];
	float mean[3];
};


static void skitrack1_process (struct skitrack1 * s)
{
	ASSERT_PARAM_NOTNULL (s);

	//Remove bad points:
	point_filter (s->pc, POINT_STRIDE, s->pc, POINT_STRIDE, &s->pc_count, 3, 1.0f);

	//Move the center of all points to origin:
	vf32_move_center_to_zero (DIMENSION (3), s->pc, POINT_STRIDE, s->pc_count, s->mean);

	//Calculate the covariance matrix of the points which can be used to get the orientation of the points:
	mf32_get_covariance (DIMENSION (3), s->pc, POINT_STRIDE, s->pc_count, s->c);

	//Calculate the eigen vectors (c) and eigen values (w) from covariance matrix (c) which will get the orientation of the points:
	//https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dsyev.htm
	LAPACKE_ssyev (LAPACK_COL_MAJOR, 'V', 'U', DIMENSION (3), s->c, DIMENSION (3), s->w);

	//LAPACK_ssyev ();
	printf ("eigen vector:\n"); m3f32_print (s->c, stdout);
	printf ("eigen value: %f %f %f\n", s->w[0], s->w[1], s->w[2]);

	//Rectify every point by this rotation matrix which is the current orientation of the points:
	s->crot[0] = s->c[3];
	s->crot[1] = s->c[6];
	s->crot[2] = s->c[0];

	s->crot[3] = s->c[4];
	s->crot[4] = s->c[7];
	s->crot[5] = s->c[1];

	s->crot[6] = s->c[5];
	s->crot[7] = s->c[8];
	s->crot[8] = s->c[2];
	//TODO: Do a matrix matrix multiplication instead of matrix vector multiplication:
	for (uint32_t i = 0; i < s->pc_count; ++i)
	{
		float * v = s->pc + (i * POINT_STRIDE);
		mv3f32_mul (v, s->crot, v);
	}
	//cblas_sgemm (CblasColMajor, CblasTrans, CblasNoTrans, 4, point_pos1_count, 4, 1.0f, rotation, 4, point_pos1, 4, 0.0f, point_pos1, 4);
}



#define SKITRACK2_PEAKS_COUNT 2

struct skitrack2
{
	float img1[IMG_XN*IMG_YN];//Projected points
	float img2[IMG_XN*IMG_YN];//Convolution from img1
	float img3[IMG_XN*IMG_YN];//Convolution from img2
	float imgf[IMG_XN*IMG_YN];//Used for normalizing pixel
	float q1[IMG_YN];
	float q2[IMG_YN];
	uint32_t g[SKITRACK2_PEAKS_COUNT];
	float k;
};



static void skitrack2_process (struct skitrack2 * s, float pc[], uint32_t pc_count)
{
	ASSERT_PARAM_NOTNULL (s);

	//Project 3D points to a 2D image:
	//The center of the image is put ontop of the origin where all points are:
	point_project (s->img1, s->imgf, IMG_XN, IMG_YN, pc, POINT_STRIDE, pc_count);


	//Amplify skitrack pattern in the 2D image:
	{
		int32_t kxn = 1;
		int32_t kyn = 5;
		float kernel[1*5] =
		{
		-5.0f,
		2.0f,
		6.0f,
		2.0f,
		-5.0f
		};
		vf32_normalize (kxn*kyn, kernel, kernel);
		vf32_convolution2d (s->img2, s->img1, IMG_XN, IMG_YN, kernel, kxn, kyn);
	}



	//vf32_remove_low_values (img2, IMG_XN*IMG_YN);


	//Smooth filter:
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
		vf32_convolution2d (s->img3, s->img2, IMG_XN, IMG_YN, kernel, kxn, kyn);
	}


	//vf32_remove_low_values (img3, IMG_XN*IMG_YN);
	//memcpy (img3, img2, sizeof(img3));



	//Find the most common lines direction in the image which hopefully matches the direction of the skitrack:
	//Project 2D image to a 1D image in the the most common direction (k):

	//float k = vf32_most_common_line (img3, IMG_XN, IMG_YN, 20);
	s->k = vf32_most_common_line2 (s->img3, IMG_XN, IMG_YN, s->q1);
	//vf32_project_2d_to_1d (img3, IMG_XN, IMG_YN, k, q1);
	vf32_project_2d_to_1d_pn (s->img3, IMG_XN, IMG_YN, s->k, s->q1);
	vf32_remove_low_values (s->q1, IMG_YN);


	//Amplify skitrack pattern in the 1D image:
	float skitrack_kernel1d[] =
	{
	 1.0f,  3.0f,  1.0f,
	-3.0f, -9.0f, -3.0f,
	 1.0f,  7.0f,  1.0f,
	-3.0f, -9.0f, -3.0f,
	 1.0f,  3.0f,  1.0f
	};
	vf32_convolution1d (s->q1, IMG_YN, s->q2, skitrack_kernel1d, countof (skitrack_kernel1d));


	//Find the peaks which should be where the skitrack is positioned:
	{
		float q[IMG_YN] = {0.0f};
		memcpy (q, s->q2, sizeof (q));
		vf32_find_peaks (q, IMG_YN, s->g, SKITRACK2_PEAKS_COUNT, 16, 20);
	}


	//vf32_normalize (countof (q1), q1, q1);
	//vf32_normalize (countof (q2), q2, q2);

}



static void points_test_sinus_slope (float points[])
{
	//Test sinus slopes:
	for (float * p = points; p < points + LIDAR_WH*POINT_STRIDE; p += POINT_STRIDE)
	{
		p[2] += 0.6f * sin (1.0f*p[1]);
	}
}




