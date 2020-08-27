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



enum visual_line
{
	VISUAL_LINE_ORIGIN_0,
	VISUAL_LINE_ORIGIN_1,
	VISUAL_LINE_ORIGIN_2,
	VISUAL_LINE_PCA_0,
	VISUAL_LINE_PCA_1,
	VISUAL_LINE_PCA_2,
	VISUAL_LINE_SKITRACK,
	VISUAL_LINE_SKITRACK_END = VISUAL_LINE_SKITRACK + SKITRACK2_PEAKS_COUNT - 1,
	VISUAL_LINE_COUNT
};









/*
	 1: Read filename                   : (Filename) -> (text 3D points)
	 2: Convert text points to f32      : (text 3D points) -> (3D points)
	 3: Filter out bad points           : (3D points) -> (3D points)
	 4: (PCA) Move center to origin     : (3D points) -> (3D points)
	 5: (PCA) Get covariance matrix     : (3D points) -> (3x3 matrix)
	 6: (PCA) Get eigen vectors         : (3x3 matrix) -> (3x3 rotation matrix)
	 7: (PCA) Rectify points            : ((3x3 rotation matrix), (3D points)) -> (3D points)
	 8: Project 3D points to 2D image   : (3D points)) -> (2D image)
	 9: (Conv) Amplify skitrack         : (2D image) -> (2D image)
	10: Remove low values               : (2D image) -> (2D image)
	11: (Conv) Smooth                   : (2D image) -> (2D image)
	12: Find most common line direction : (2D image) -> (direction)
	13: Project 2D image to 1D image    : ((2D image), (direction)) -> (1D image)
	14: Remove low values               : (1D image) -> (1D image)
	15: (Conv) Amplify 1D skitracks     : (1D image) -> (1D image)
	16: Find all peaks                  : (1D image) -> ((position), (strength))
	17: Output of skitrack position     : ((position), (strength))
*/
void show (const char * filename, nng_socket socks[])
{
	struct skitrack1 s1 = {0};
	struct skitrack2 s2 = {0};

	float pointpos[LIDAR_WH*POINT_STRIDE*2];
	uint32_t pointcol[LIDAR_WH*2] = {RGBA (0xFF, 0xFF, 0xFF, 0xFF)};//The color of each point. This is only used for visualization.
	uint32_t imgv[IMG_XN*IMG_YN] = {0};//Used for visual confirmation that the algorithm works


	points_read_filename (filename, s1.pc, &s1.pc_count);
	points_test_sinus_slope (s1.pc);

	memcpy (pointpos, s1.pc, LIDAR_WH*POINT_STRIDE*sizeof(float));
	skitrack1_process (&s1);
	skitrack2_process (&s2, s1.pc, s1.pc_count);
	memcpy (pointpos + LIDAR_WH*POINT_STRIDE, s1.pc, LIDAR_WH*POINT_STRIDE*sizeof(float));

	//Visualize the skitrack and more information:
	image_visual (imgv, s2.img3, IMG_XN, IMG_YN, s2.q1, s2.q2, s2.g, SKITRACK2_PEAKS_COUNT, s2.k);




	//pix_rgba[105*IMG_XN + 12] |= RGBA(0x00, 0x66, 0x00, 0x00);
	//pix_rgba[0*IMG_XN + 1] |= RGBA(0x00, 0xFF, 0x00, 0xFF);
	//pix_rgba[2*IMG_XN + 0] |= RGBA(0x00, 0xFF, 0xff, 0xFF);
	//pix_rgba[2*IMG_XN + 1] |= RGBA(0x00, 0xFF, 0xff, 0xFF);

	struct v4f32_line linepos1[VISUAL_LINE_COUNT];
	struct u32_line linecol1[VISUAL_LINE_COUNT];

	vf32_set3 (linepos1[VISUAL_LINE_ORIGIN_0].a, 0.0f, 0.0f, 0.0f);
	vf32_set3 (linepos1[VISUAL_LINE_ORIGIN_0].b, 1.0f, 0.0f, 0.0f);
	vf32_set3 (linepos1[VISUAL_LINE_ORIGIN_1].a, 0.0f, 0.0f, 0.0f);
	vf32_set3 (linepos1[VISUAL_LINE_ORIGIN_1].b, 0.0f, 1.0f, 0.0f);
	vf32_set3 (linepos1[VISUAL_LINE_ORIGIN_2].a, 0.0f, 0.0f, 0.0f);
	vf32_set3 (linepos1[VISUAL_LINE_ORIGIN_2].b, 0.0f, 0.0f, 1.0f);

	vf32_set3 (linepos1[VISUAL_LINE_PCA_0].a, 0.0f   , 0.0f   , 0.0f   );
	vf32_set3 (linepos1[VISUAL_LINE_PCA_0].b, s1.c[0], s1.c[1], s1.c[2]);
	vf32_set3 (linepos1[VISUAL_LINE_PCA_1].a, 0.0f   , 0.0f   , 0.0f   );
	vf32_set3 (linepos1[VISUAL_LINE_PCA_1].b, s1.c[3], s1.c[4], s1.c[5]);
	vf32_set3 (linepos1[VISUAL_LINE_PCA_2].a, 0.0f   , 0.0f   , 0.0f   );
	vf32_set3 (linepos1[VISUAL_LINE_PCA_2].b, s1.c[6], s1.c[7], s1.c[8]);




	linecol1[VISUAL_LINE_ORIGIN_0].a = RGBA(0xFF, 0x00, 0x00, 0xFF);
	linecol1[VISUAL_LINE_ORIGIN_0].b = RGBA(0xFF, 0x00, 0x00, 0xFF);
	linecol1[VISUAL_LINE_ORIGIN_1].a = RGBA(0x00, 0xFF, 0x00, 0xFF);
	linecol1[VISUAL_LINE_ORIGIN_1].b = RGBA(0x00, 0xFF, 0x00, 0xFF);
	linecol1[VISUAL_LINE_ORIGIN_2].a = RGBA(0x00, 0x00, 0xFF, 0xFF);
	linecol1[VISUAL_LINE_ORIGIN_2].b = RGBA(0x00, 0x00, 0xFF, 0xFF);

	linecol1[VISUAL_LINE_PCA_0].a = RGBA(0xFF, 0x00, 0x00, 0xFF) | RGBA(0xAA, 0xAA, 0xAA, 0xFF);
	linecol1[VISUAL_LINE_PCA_0].b = RGBA(0xFF, 0x00, 0x00, 0xFF) | RGBA(0xAA, 0xAA, 0xAA, 0xFF);
	linecol1[VISUAL_LINE_PCA_1].a = RGBA(0x00, 0xFF, 0x00, 0xFF) | RGBA(0xAA, 0xAA, 0xAA, 0xFF);
	linecol1[VISUAL_LINE_PCA_1].b = RGBA(0x00, 0xFF, 0x00, 0xFF) | RGBA(0xAA, 0xAA, 0xAA, 0xFF);
	linecol1[VISUAL_LINE_PCA_2].a = RGBA(0x00, 0x00, 0xFF, 0xFF) | RGBA(0xAA, 0xAA, 0xAA, 0xFF);
	linecol1[VISUAL_LINE_PCA_2].b = RGBA(0x00, 0x00, 0xFF, 0xFF) | RGBA(0xAA, 0xAA, 0xAA, 0xFF);

	for (int i = VISUAL_LINE_SKITRACK; i <= VISUAL_LINE_SKITRACK_END; ++i)
	{
		linecol1[i].a = RGBA(0x77, 0xFF, 0x11, 0xFF);
		linecol1[i].b = RGBA(0x77, 0xFF, 0x11, 0xFF);
	}

	{
		pixel_to_point (linepos1[VISUAL_LINE_SKITRACK+0].a, IMG_XN, IMG_YN, 0.0f, -10.0f, s2.g[0] - 10.0f * s2.k);
		pixel_to_point (linepos1[VISUAL_LINE_SKITRACK+0].b, IMG_XN, IMG_YN, 0.0f,  30.0f, s2.g[0] + 30.0f * s2.k);
		pixel_to_point (linepos1[VISUAL_LINE_SKITRACK+1].a, IMG_XN, IMG_YN, 0.0f, -10.0f, s2.g[1] - 10.0f * s2.k);
		pixel_to_point (linepos1[VISUAL_LINE_SKITRACK+1].b, IMG_XN, IMG_YN, 0.0f,  30.0f, s2.g[1] + 30.0f * s2.k);
		float rot[3*3];
		memcpy (rot, s1.crot, sizeof (rot));
		m3f32_lapacke_inverse (rot, 3);
		for (float * i = linepos1[VISUAL_LINE_SKITRACK+0].a; i <= linepos1[VISUAL_LINE_SKITRACK_END].b; i += POINT_STRIDE)
		{
			mv3f32_mul (i, rot, i);
			vvf32_add (4, i, i, s1.mean);
		}
	}



	//Send visual information to the graphic server:
	{
		int r;
		r = nng_send (socks[MAIN_NNGSOCK_LINE_POS], linepos1, VISUAL_LINE_COUNT*POINT_STRIDE*2*sizeof(float), 0);
		perror (nng_strerror (r));
		r = nng_send (socks[MAIN_NNGSOCK_LINE_COL], linecol1, VISUAL_LINE_COUNT*2*sizeof(uint32_t), 0);
		perror (nng_strerror (r));
		r = nng_send (socks[MAIN_NNGSOCK_POINTCLOUD_POS], pointpos, LIDAR_WH*POINT_STRIDE*sizeof(float)*2, 0);
		perror (nng_strerror (r));
		r = nng_send (socks[MAIN_NNGSOCK_POINTCLOUD_COL], pointcol, LIDAR_WH*sizeof(uint32_t)*2, 0);
		perror (nng_strerror (r));
		r = nng_send (socks[MAIN_NNGSOCK_TEX], imgv, IMG_XN*IMG_YN*sizeof(uint32_t), 0);
		perror (nng_strerror (r));
	}

}



int main (int argc, char const * argv[])
{
	ASSERT (argc);
	ASSERT (argv);

#ifdef USING_QT_CREATOR
	chdir ("../ce30_demo/");
#endif

	csc_crossos_enable_ansi_color();

	nng_socket socks[MAIN_NNGSOCK_COUNT] = {{0}};
	main_nng_pairdial (socks + MAIN_NNGSOCK_POINTCLOUD_POS, "tcp://localhost:9002");
	main_nng_pairdial (socks + MAIN_NNGSOCK_POINTCLOUD_COL, "tcp://localhost:9003");
	main_nng_pairdial (socks + MAIN_NNGSOCK_TEX,            "tcp://localhost:9004");
	main_nng_pairdial (socks + MAIN_NNGSOCK_VOXEL,          "tcp://localhost:9005");
	main_nng_pairdial (socks + MAIN_NNGSOCK_LINE_POS,       "tcp://localhost:9006");
	main_nng_pairdial (socks + MAIN_NNGSOCK_LINE_COL,       "tcp://localhost:9007");

	chdir ("txtpoints/1");
	show ("14_13_57_24145.txt", socks);
	//show ("14_13_55_22538.txt", socks);
	//show ("14_13_53_20565.txt", socks);
	//show ("14_13_52_19801.txt", socks);
	//show ("14_13_53_20906.txt", socks);
	//show ("14_13_55_22978.txt", socks);
	//show ("14_13_58_25517.txt", socks);
	//show ("14_13_53_20783.txt", socks);
	//show ("14_13_55_22978.txt", socks);
	//show ("14_13_54_21339.txt", socks);
	//show ("14_13_59_26063.txt", socks);
	//show ("14_16_57_204577.txt", socks);
	//return 0;

#if 0
	FILE * f = popen ("ls", "r");
	ASSERT (f);
	char buf[200] = {'\0'};
	while (1)
	{
		int c = getchar();
		switch (c)
		{
		case 'q':
			goto exit_while;
			break;
		case '\n':
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
