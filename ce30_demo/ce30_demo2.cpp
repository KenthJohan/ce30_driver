#include <iostream>
#include <ce30_driver/ce30_driver.h>

using namespace std;
using namespace ce30_driver;


int main()
{
	UDPSocket socket;
	if (socket.Connect() != Diagnose::connect_successful)
	{
		return -1;
	}
	Packet packet;
	Scan scan;
	while (true)
	{
		if (!GetPacket(packet, socket)) {
			continue;
		}
		unique_ptr<ParsedPacket> parsed = packet.Parse();
		if (parsed)
		{
			scan.AddColumnsFromPacket(*parsed);
			if (!scan.Ready())
			{
				continue;
			}
			for (int x = 0; x < scan.Width(); ++x)
			{
				for (int y = 0; y < scan.Height(); ++y)
				{
					Channel channel = scan.at(x, y);
					cout <<
					"(" << channel.distance << ", " << channel.amplitude << ") "
					"[" <<
					channel.point().x << ", " <<
					channel.point().y << ", " <<
					channel.point().z << "]" << endl;
				}
			}
			scan.Reset();
		}
	}
}
