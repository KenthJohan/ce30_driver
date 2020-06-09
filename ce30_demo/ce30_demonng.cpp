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

using namespace std;
using namespace ce30_driver;

#define POINTC_W 320
#define POINTC_H 20


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




struct vplane
{
	float a[4];
	float b[4];
	float c[4];
	float d[4];
	float e[4];
	float f[4];
};


static void send_plane (nng_socket sock, float plane[4])
{
	float r = 0.01f;
	float s = 0.01f;
	struct vplane v =
	{
	{0.0f, -r, -s, 0.0f},   //left,  bottom
	{0.0f,  r, -s, 0.0f},    //right, bottom
	{0.0f,  r,  s, 0.0f},   //right, top
	{0.0f, -r, -s, 0.0f},   //left,  bottom
	{0.0f,  r,  s, 0.0f},   //right, top
	{0.0f, -r,  s, 0.0f},   //left,  top
};
	csc_linmat_plane_to_parametric (v.a, plane);
	csc_linmat_plane_to_parametric (v.b, plane);
	csc_linmat_plane_to_parametric (v.c, plane);
	csc_linmat_plane_to_parametric (v.d, plane);
	csc_linmat_plane_to_parametric (v.e, plane);
	csc_linmat_plane_to_parametric (v.f, plane);
	main_nng_send (sock, (float*)&v, 4*sizeof(float)*6);
}



static void crosser (nng_socket sock, nng_socket sockq, float w[], unsigned n)
{
	static float avg[3] = {0.0f};
	float avg1[3] = {0.0f};
	float avg2[3] = {0.0f};
	float up[3] = {0.0f, 1.0f, 0.0f};
	unsigned i = rand() % n;
	unsigned j = rand() % n;
	unsigned k = rand() % n;
	if (i == j) {return;}
	if (i == k) {return;}
	float * a = w + i*4;
	float * b = w + j*4;
	float * c = w + k*4;
	float ab[3];
	float ac[3];
	float az[3];
	float z[3];
	float v[12*4];
	vvf32_sub (3, ab, a, b);
	vvf32_sub (3, ac, a, c);
	v3f32_cross (az, ab, ac);
	//az[0] = fabs(az[0]);
	//az[1] = fabs(az[1]);
	//az[2] = fabs(az[2]);
	//printf ("vvf32_dot (3, az, up) %f\n", vvf32_dot (3, az, up));
	if (vvf32_dot (3, az, up) < 0)
	{
		az[0] = -az[0];
		az[1] = -az[1];
		az[2] = -az[2];
	};
	vf32_normalize (3, az, az);
	vvf32_add (3, avg, avg, az);
	vf32_normalize (3, avg1, avg);
	vsf32_mul (3, az, az, 5.0f);
	vvf32_add (3, z, a, az);
	vsf32_mul (3, avg1, avg1, 5.0f);
	vf32_cpy (3, v + 9*4, avg1);
	vvf32_add (3, avg1, a, avg1);
	vf32_cpy (3, v + 0*4, a);
	vf32_cpy (3, v + 1*4, b);
	vf32_cpy (3, v + 2*4, a);
	vf32_cpy (3, v + 3*4, c);
	vf32_cpy (3, v + 4*4, a);
	vf32_cpy (3, v + 5*4, z);
	vf32_cpy (3, v + 6*4, a);
	vf32_cpy (3, v + 7*4, avg1);

	v[8*4 + 0] = 0.0f;
	v[8*4 + 1] = 0.0f;
	v[8*4 + 2] = 0.0f;

	v[10*4 + 0] = 0.0f;
	v[10*4 + 1] = 0.0f;
	v[10*4 + 2] = 0.0f;
	v[11*4 + 0] = 0.0f;
	v[11*4 + 1] = 5.0f;
	v[11*4 + 2] = 0.0f;

	main_nng_send (sock, v, 4*sizeof(float)*12);
	//float q[4];
	//qf32_xyza (q, az[0], az[2], az[1], M_PI);
	//qf32_normalize (q, q);
	//printf ("%+1.4f %+1.4f %+1.4f %+1.4f\n", q[0], q[1], q[2], q[3]);
	//send123 (sockq, q, 1);
}


static void find_plane (float x[], unsigned n, float min[4], float max[4], nng_socket sock)
{
	float v[4*6];
	for (int i = 0; i < n; ++i)
	{
		min[0] = MIN(min[0], x[0]);
		min[1] = MIN(min[1], x[1]);
		min[2] = MIN(min[2], x[2]);
		max[0] = MAX(max[0], x[0]);
		max[1] = MAX(max[1], x[1]);
		max[2] = MAX(max[2], x[2]);
		x += 4;
	}
	float v1[4] = {min[0], min[1], 0.0f, 1.0f}; //left,  bottom
	float v2[4] = {max[0], min[1], 0.0f, 1.0f}; //right, bottom
	float v3[4] = {max[0], max[1], 0.0f, 1.0f}; //right, top
	float v4[4] = {min[0], min[1], 0.0f, 1.0f}; //left,  bottom
	float v5[4] = {max[0], max[1], 0.0f, 1.0f}; //right, top
	float v6[4] = {min[0], max[1], 0.0f, 1.0f}; //left,  top
	vf32_cpy (4, v+0*4, v1);
	vf32_cpy (4, v+1*4, v2);
	vf32_cpy (4, v+2*4, v3);
	vf32_cpy (4, v+3*4, v4);
	vf32_cpy (4, v+4*4, v5);
	vf32_cpy (4, v+5*4, v6);
	main_nng_send (sock, v, 4*sizeof(float)*6);
}


static void convert_points (float points[], Scan const &scan)
{
	ASSERT (POINTC_W == scan.Width());
	ASSERT (POINTC_H == scan.Height());
	float * p = points;
	for (int x = 0; x < POINTC_W; ++x)
	{
		for (int y = 0; y < POINTC_H; ++y)
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


static void ransac (nng_socket sock, float const p[], unsigned n, float plane[4])
{
	unsigned i = rand() % n;
	unsigned j = i + (rand() & 0xFF);
	unsigned k = j + (rand() & 0xFF);
	if (i == j) {return;}
	if (i == k) {return;}
	float const * a = p + i*4;
	float const * b = p + j*4;
	float const * c = p + k*4;
	csc_linmat_plane_from_3points (plane, a, b, c);
	if (vf32_sum(3, plane) > 0.0f)
	{
		printf ("abc (%f %f %f) (%f %f %f) (%f %f %f)\n", a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
		printf ("ijk %i %i %i\n", i, j, k);
		printf ("plane %f %f %f %f\n", plane[0], plane[1], plane[2], plane[3]);
		unsigned q = csc_linmat_plane_point_count (plane, p, n, 0.001f);
		printf ("q %i\n", q);
		send_plane (sock, plane);
	}
}

/*
struct particle
{
	float a[4];
	float b[4];
	float c[4];
};


void particle_force (struct particle p[], struct particle v[])
{
	vvf32_add (3, p->a, p->a, v->a);
	vvf32_add (3, p->b, p->b, v->b);
	vvf32_add (3, p->c, p->c, v->c);
	float ab[4];
	float ac[4];
	float bc[4];
	vvf32_sub (3, ab, p->b, p->a);
	vvf32_sub (3, ac, p->c, p->a);
	vvf32_sub (3, bc, p->c, p->b);

	vvf32_add (3, v->a, v->a, ab);
	vvf32_add (3, v->a, v->a, ac);

	vvf32_add (3, v->b, v->b, bc);
	vvf32_add (3, v->b, v->b, ab);

	vvf32_add (3, v->c, v->c, ab);
}


static void particle_method (nng_socket sock, float const p[], unsigned n, struct particle * particle)
{

}
*/


#define VOX_XN 50
#define VOX_YN 20
#define VOX_ZN 10
#define VOX_I(x,y,z) ((z)*VOX_XN*VOX_YN + (y)*VOX_XN + (x))
#define VOX_SCALE 0.1f



static void main_vox_neighbor (uint8_t v[], uint8_t x, uint8_t y, uint8_t z)
{
	static uint8_t id = 0;
	uint32_t i = VOX_I(x, y, z);
	uint32_t i000 = v[VOX_I(x-1, y-1, z-1)];
	uint32_t i001 = v[VOX_I(x-1, y-1, z+1)];
	uint32_t i010 = v[VOX_I(x-1, y+1, z-1)];
	uint32_t i011 = v[VOX_I(x-1, y+1, z+1)];
	uint32_t i100 = v[VOX_I(x+1, y-1, z-1)];
	uint32_t i101 = v[VOX_I(x+1, y-1, z+1)];
	uint32_t i110 = v[VOX_I(x+1, y+1, z-1)];
	uint32_t i111 = v[VOX_I(x+1, y+1, z+1)];
	v[i] = 0;
	if (0){}
	else if (v[i] == v[i000]){v[i] = v[i000];}
	else if (v[i] == v[i001]){v[i] = v[i001];}
	else if (v[i] == v[i010]){v[i] = v[i010];}
	else if (v[i] == v[i011]){v[i] = v[i011];}
	else if (v[i] == v[i100]){v[i] = v[i100];}
	else if (v[i] == v[i101]){v[i] = v[i101];}
	else if (v[i] == v[i110]){v[i] = v[i110];}
	else if (v[i] == v[i111]){v[i] = v[i111];}
	else if (v[i] == 0)
	{
		id++;
		v[i] = id;
	}
}

static void main_test_voxels (nng_socket sock, uint8_t voxel[VOX_XN*VOX_YN*VOX_ZN], float const p[], unsigned n)
{
	voxel[VOX_I(0,0,0)] = 122;
	for (unsigned i = 0; i < n; ++i, p+=4)
	{
		float fx = (p[0])/VOX_SCALE;
		float fy = (p[1])/VOX_SCALE;
		float fz = (p[2])/VOX_SCALE;
		uint8_t ux = fx;
		uint8_t uy = fy+VOX_YN/2;
		uint8_t uz = fz+VOX_ZN/2;
		if (ux >= VOX_XN){continue;}
		if (uy >= VOX_YN){continue;}
		if (uz >= VOX_ZN){continue;}
		if (ux <= 0){continue;}
		if (uy <= 0){continue;}
		if (uz <= 0){continue;}
		main_vox_neighbor (voxel, ux, uy, uz);
	}
	main_nng_send (sock, voxel, VOX_XN*VOX_YN*VOX_ZN);
}





enum main_nngsock
{
	MAIN_NNGSOCK_POINTCLOUD,
	MAIN_NNGSOCK_PLANE,
	MAIN_NNGSOCK_TEX,
	MAIN_NNGSOCK_VOXEL,
	MAIN_NNGSOCK_COUNT
};


int main()
{
	csc_crossos_enable_ansi_color();

	nng_socket socks[MAIN_NNGSOCK_COUNT] = {{0}};
	main_nng_pairdial (socks + MAIN_NNGSOCK_POINTCLOUD, "tcp://192.168.1.176:9002");
	main_nng_pairdial (socks + MAIN_NNGSOCK_VOXEL, "tcp://192.168.1.176:9005");



	float points[POINTC_W*POINTC_H*4] = {0.0f};
	uint8_t voxel[VOX_XN*VOX_YN*VOX_ZN] = {0};

	random_points (points, POINTC_W*POINTC_H);
	main_nng_send (socks[MAIN_NNGSOCK_POINTCLOUD], points, POINTC_W*POINTC_H*4*sizeof(float));

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
		convert_points (points, scan);
		scan.Reset();

		//float points_min[4] = {FLT_MAX};
		//float points_max[4] = {FLT_MIN};
		//find_plane (points, POINTC_W*POINTC_H, points_min, points_max, socket_planefit);

		//float plane[4];
		//ransac (socket_planefit, points, POINTC_W*POINTC_H, plane);


		main_nng_send (socks[MAIN_NNGSOCK_POINTCLOUD], points, POINTC_W*POINTC_H*4*sizeof(float));
		main_test_voxels (socks[MAIN_NNGSOCK_VOXEL], voxel, points, POINTC_W*POINTC_H);
	}
}
