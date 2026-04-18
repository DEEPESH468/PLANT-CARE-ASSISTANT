# Packet Journey Visualizer

Computer Networks PBL project for demonstrating how a browser request becomes network communication.

## Project Topic

**DNS Resolution and TCP Connection Simulation for Web Browsing**

This project explains this flow:

```text
Source IP: 192.168.1.100 -> Dest IP: 8.8.8.8
Protocol: DNS | Port: 53 | Payload: Query for google.com

Source IP: 192.168.1.100 -> Dest IP: 142.250.185.46
Protocol: TCP | Port: 443 | Status: SYN
```

## Features

- Resolves a domain name using DNS.
- Shows DNS query and DNS response packets.
- Simulates the TCP 3-way handshake: SYN, SYN-ACK, ACK.
- Shows HTTPS connection setup on port 443.
- Includes browser UI, local DNS API mode, optional Tkinter GUI mode, and CLI mode.
- Uses only Python standard library modules.

## How To Run

From this folder:

```bash
python3 app.py --web
```

Then open the URL shown in the terminal, usually:

```bash
http://127.0.0.1:8000/
```

This mode uses Python's `socket` module for live DNS lookup, so different websites can return different real IP addresses.

You can still open the static browser file:

```bash
open index.html
```

But static browser mode cannot do live DNS by itself. It uses demo fallback IP values if the local Python server is not running.

CLI mode:

```bash
python3 app.py --cli --domain google.com
```

Multiple domains or URLs:

```bash
python3 app.py --cli --domain "google.com, github.com, https://openai.com"
```

If your environment has a virtual environment:

```bash
venv/bin/python app.py --web
venv/bin/python app.py --cli --domain google.com
```

## Concepts Covered

- IP addressing
- DNS
- Client-server communication
- TCP port numbers
- TCP 3-way handshake
- HTTPS on port 443

## Viva Explanation

When the user enters `google.com`, the client first sends a DNS query to DNS server `8.8.8.8` on port `53`. The DNS server returns an IP address for the website. After that, the client starts a TCP connection to the web server on port `443`. TCP uses a 3-way handshake: first `SYN`, then `SYN-ACK`, then `ACK`. After the handshake, the browser is ready to communicate with the website using HTTPS.

## Sample Output

```text
No. | Source IP     | Destination IP | Protocol | Port | Status         | Payload
1   | 192.168.1.100 | 8.8.8.8        | DNS      | 53   | Query          | Client asks DNS server for google.com
2   | 8.8.8.8       | 192.168.1.100  | DNS      | 53   | Response       | DNS server returns 142.250.185.46
3   | 192.168.1.100 | 142.250.185.46 | TCP      | 443  | SYN            | Client starts TCP 3-way handshake
4   | 142.250.185.46| 192.168.1.100  | TCP      | 443  | SYN-ACK        | Server accepts and acknowledges the connection request
5   | 192.168.1.100 | 142.250.185.46 | TCP      | 443  | ACK            | Client acknowledges the server response
6   | 192.168.1.100 | 142.250.185.46 | HTTPS    | 443  | Secure Request | Browser is ready to request https://google.com
```
