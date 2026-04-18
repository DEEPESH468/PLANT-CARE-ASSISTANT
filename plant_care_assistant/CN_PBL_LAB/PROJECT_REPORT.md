# PBL Project Report

## Title

DNS Resolution and TCP Connection Simulation for Web Browsing

## Subject

Computer Networks

## Objective

The objective of this project is to demonstrate how a normal website request travels through a computer network. The project shows how a domain name is converted into an IP address using DNS and how the client creates a TCP connection with the web server using the TCP 3-way handshake.

## Problem Statement

Users type website names such as `google.com`, but computers communicate using IP addresses. This project explains what happens between entering a domain name and starting a secure web connection.

## Technologies Used

- Python 3
- Tkinter for GUI
- Socket module for DNS lookup
- Dataclasses for packet structure

## Network Concepts

### DNS

DNS stands for Domain Name System. It converts a human-readable domain name such as `google.com` into a machine-readable IP address such as `142.250.185.46`.

### IP Address

An IP address identifies a device or server on a network. In this project, the client IP is simulated as `192.168.1.100`.

### Port Number

A port number identifies a service running on a device. DNS commonly uses port `53`, and HTTPS commonly uses port `443`.

### TCP 3-Way Handshake

TCP creates a reliable connection before data transfer. The handshake has three steps:

1. SYN: Client requests a connection.
2. SYN-ACK: Server accepts and acknowledges the request.
3. ACK: Client confirms the connection.

## Project Workflow

1. The user enters a domain name.
2. The program performs a DNS lookup using Python's socket module.
3. The program displays the DNS query packet.
4. The program displays the DNS response packet.
5. The program simulates TCP SYN, SYN-ACK, and ACK packets.
6. The program shows that HTTPS communication is ready on TCP port `443`.

## Example Packet Flow

```text
Client IP: 192.168.1.100
DNS Server: 8.8.8.8
Domain: google.com
Web Server IP: 142.250.185.46
```

```text
1. 192.168.1.100 -> 8.8.8.8        DNS    Port 53   Query google.com
2. 8.8.8.8       -> 192.168.1.100  DNS    Port 53   Response with server IP
3. 192.168.1.100 -> Web Server     TCP    Port 443  SYN
4. Web Server    -> 192.168.1.100  TCP    Port 443  SYN-ACK
5. 192.168.1.100 -> Web Server     TCP    Port 443  ACK
6. 192.168.1.100 -> Web Server     HTTPS  Port 443  Secure request ready
```

## Expected Outcome

The project helps students understand the journey of packets in a web browsing request. It also makes the difference between DNS, TCP, ports, and HTTPS easier to explain during a viva or demonstration.

## Future Scope

- Add real packet capture using Scapy.
- Add packet filtering by protocol.
- Add animation for packet movement.
- Export packet logs to CSV.
