#include <iostream>
#include <chrono>
#include <ce30_driver/ce30_driver.h>


using namespace std;
using namespace ce30_driver;

#define POINTC_W 320
#define POINTC_H 20




int main()
{
	float points[POINTC_W*POINTC_H*4] = {0.0f};
	UDPSocket socket;
	if (socket.Connect() != Diagnose::connect_successful)
	{
		return -1;
	}
	Packet packet;
	Scan scan;
	printf ("Loop:\n");
	float sum = 0.0f;
	float sum_old = 0.0f;

	std::chrono::duration<float, std::milli> elapsed_sum;
	unsigned number_of_iterations = 1000;
	for (unsigned i = 0; i < number_of_iterations; ++i)
	{
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		if (!GetPacket (packet, socket)){continue;}
		unique_ptr<ParsedPacket> parsed = packet.Parse();
		if (!parsed){continue;}
		scan.AddColumnsFromPacket (*parsed);
		if (!scan.Ready()){continue;}
		float * p = points;
		sum = 0.0f;
		for (int x = 0; x < scan.Width(); ++x)
		{
			for (int y = 0; y < scan.Height(); ++y)
			{
				Channel channel = scan.at(x, y);
				p[0] = channel.point().x;
				p[2] = channel.point().y;
				p[1] = channel.point().z;
				p[3] = 1.0f;
				p += 4;
				sum += p[0] + p[1] + p[2] + p[3];
			}
		}
		uint32_t x;
		memcpy (&x, &sum, sizeof (float));
		printf ("Checksum: %016x\n", x);
		if (sum == sum_old){continue;}
		sum_old = sum;
		scan.Reset();
		std::chrono::duration<float, std::milli> elapsed = std::chrono::high_resolution_clock::now() - start;
		elapsed_sum += elapsed;
		printf ("millisecond ticks %6.2f\n", elapsed.count());
	}
	printf ("average millisecond ticks %f\n", elapsed_sum.count() / (float)number_of_iterations);
}
