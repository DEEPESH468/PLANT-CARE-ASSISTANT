#!/usr/bin/env python3
"""DNS and TCP packet journey visualizer for a Computer Networks PBL project."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import webbrowser
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlparse

try:
    import tkinter as tk
    from tkinter import ttk
except ModuleNotFoundError:
    tk = None
    ttk = None


CLIENT_IP = "192.168.1.100"
DNS_SERVER_IP = "8.8.8.8"
DEFAULT_DOMAIN = "google.com"
FALLBACK_IPS = {
    "google.com": "142.250.185.46",
    "www.google.com": "142.250.185.46",
    "example.com": "93.184.216.34",
    "github.com": "140.82.112.3",
    "openai.com": "104.18.33.45",
    "youtube.com": "142.250.185.46",
}

PROJECT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Packet:
    number: int
    source: str
    destination: str
    protocol: str
    port: int | str
    status: str
    payload: str


def normalize_domain(value: str) -> str:
    """Accept a domain or full URL and return only the host name."""
    cleaned = value.strip()
    if not cleaned:
        return DEFAULT_DOMAIN

    if "://" not in cleaned:
        cleaned = "https://" + cleaned

    parsed = urlparse(cleaned)
    host = parsed.hostname or value.strip()
    return host.strip().lower().rstrip(".") or DEFAULT_DOMAIN


def split_domain_inputs(value: str) -> list[str]:
    """Split comma/newline/space separated domain or URL inputs."""
    separators = [",", "\n", "\t"]
    cleaned = value
    for separator in separators:
        cleaned = cleaned.replace(separator, " ")

    domains: list[str] = []
    for item in cleaned.split():
        domain = normalize_domain(item)
        if domain not in domains:
            domains.append(domain)
    return domains or [DEFAULT_DOMAIN]


def resolve_domain(domain: str) -> tuple[str, bool, str]:
    """Resolve a domain name and return IP, whether it is live, and a note."""
    clean_domain = normalize_domain(domain)

    try:
        _hostname, _aliases, addresses = socket.gethostbyname_ex(clean_domain)
        if addresses:
            return addresses[0], True, "Live DNS lookup completed successfully."
    except socket.gaierror as exc:
        fallback = FALLBACK_IPS.get(clean_domain)
        if fallback:
            return fallback, False, f"DNS lookup failed, so demo fallback IP was used: {exc}."
        return "0.0.0.0", False, f"DNS lookup failed: {exc}."

    fallback = FALLBACK_IPS.get(clean_domain, "0.0.0.0")
    return fallback, False, "No DNS address returned, so demo fallback IP was used."


def build_packets(domain: str, server_ip: str) -> list[Packet]:
    """Build the packet flow for DNS resolution, TCP handshake, and HTTPS setup."""
    clean_domain = normalize_domain(domain)
    return [
        Packet(
            1,
            CLIENT_IP,
            DNS_SERVER_IP,
            "DNS",
            53,
            "Query",
            f"Client asks DNS server for {clean_domain}",
        ),
        Packet(
            2,
            DNS_SERVER_IP,
            CLIENT_IP,
            "DNS",
            53,
            "Response",
            f"DNS server returns {server_ip}",
        ),
        Packet(
            3,
            CLIENT_IP,
            server_ip,
            "TCP",
            443,
            "SYN",
            "Client starts TCP 3-way handshake",
        ),
        Packet(
            4,
            server_ip,
            CLIENT_IP,
            "TCP",
            443,
            "SYN-ACK",
            "Server accepts and acknowledges the connection request",
        ),
        Packet(
            5,
            CLIENT_IP,
            server_ip,
            "TCP",
            443,
            "ACK",
            "Client acknowledges the server response",
        ),
        Packet(
            6,
            CLIENT_IP,
            server_ip,
            "HTTPS",
            443,
            "Secure Request",
            f"Browser is ready to request https://{clean_domain}",
        ),
    ]


def packet_to_dict(packet: Packet) -> dict[str, int | str]:
    return {
        "number": packet.number,
        "source": packet.source,
        "destination": packet.destination,
        "protocol": packet.protocol,
        "port": packet.port,
        "status": packet.status,
        "payload": packet.payload,
    }


def build_resolution_result(domain: str) -> dict[str, object]:
    clean_domain = normalize_domain(domain)
    server_ip, is_live, note = resolve_domain(clean_domain)
    packets = build_packets(clean_domain, server_ip)
    return {
        "input": domain,
        "domain": clean_domain,
        "ip": server_ip,
        "isLive": is_live,
        "note": note,
        "packets": [packet_to_dict(packet) for packet in packets],
    }


def render_cli(domain: str) -> int:
    results = [build_resolution_result(item) for item in split_domain_inputs(domain)]

    for index, result in enumerate(results):
        if index:
            print()
        packets = [
            Packet(
                int(packet["number"]),
                str(packet["source"]),
                str(packet["destination"]),
                str(packet["protocol"]),
                packet["port"],
                str(packet["status"]),
                str(packet["payload"]),
            )
            for packet in result["packets"]
        ]

        print("DNS Resolution and TCP Connection Simulation")
        print("=" * 52)
        print(f"Domain       : {result['domain']}")
        print(f"Client IP    : {CLIENT_IP}")
        print(f"DNS Server   : {DNS_SERVER_IP}")
        print(f"Web Server IP: {result['ip']}")
        print(f"Lookup Type  : {'Live DNS' if result['isLive'] else 'Demo fallback'}")
        print(f"Note         : {result['note']}")
        print()
        print(format_packet_table(packets))
        print()
        print("Flow: Client -> DNS Server -> Client -> Web Server")
    return 0


def format_packet_table(packets: Iterable[Packet]) -> str:
    headers = ["No.", "Source IP", "Destination IP", "Protocol", "Port", "Status", "Payload"]
    rows = [
        [
            str(packet.number),
            packet.source,
            packet.destination,
            packet.protocol,
            str(packet.port),
            packet.status,
            packet.payload,
        ]
        for packet in packets
    ]
    widths = [
        max(len(row[index]) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    line = " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    divider = "-+-".join("-" * width for width in widths)
    body = "\n".join(
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return f"{line}\n{divider}\n{body}"


class PacketJourneyApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Packet Journey Visualizer")
        self.root.geometry("1060x720")
        self.root.minsize(900, 620)

        self.domain_var = tk.StringVar(value=DEFAULT_DOMAIN)
        self.summary_var = tk.StringVar(value="")
        self.note_var = tk.StringVar(value="")
        self.packets: list[Packet] = []

        self._build_ui()
        self.simulate()

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f6f7fb")
        style.configure("Header.TLabel", background="#f6f7fb", foreground="#111827", font=("Arial", 22, "bold"))
        style.configure("Body.TLabel", background="#f6f7fb", foreground="#374151", font=("Arial", 11))
        style.configure("Node.TLabel", background="#ffffff", foreground="#111827", font=("Arial", 12, "bold"), padding=12)
        style.configure("Accent.TButton", font=("Arial", 11, "bold"), padding=8)
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"))
        style.configure("Treeview", rowheight=30, font=("Arial", 10))

        main = ttk.Frame(self.root, padding=20)
        main.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(main, text="DNS Resolution and TCP Connection Simulation", style="Header.TLabel")
        title.pack(anchor=tk.W)

        subtitle = ttk.Label(
            main,
            text="Enter a website name to see DNS, IP addressing, TCP handshake, ports, and HTTPS setup in one flow.",
            style="Body.TLabel",
        )
        subtitle.pack(anchor=tk.W, pady=(4, 16))

        controls = ttk.Frame(main)
        controls.pack(fill=tk.X, pady=(0, 14))

        ttk.Label(controls, text="Domain:", style="Body.TLabel").pack(side=tk.LEFT, padx=(0, 8))
        entry = ttk.Entry(controls, textvariable=self.domain_var, width=36, font=("Arial", 11))
        entry.pack(side=tk.LEFT, padx=(0, 10))
        entry.bind("<Return>", lambda _event: self.simulate())

        ttk.Button(controls, text="Simulate Packet Flow", style="Accent.TButton", command=self.simulate).pack(side=tk.LEFT)

        summary = ttk.Label(main, textvariable=self.summary_var, style="Body.TLabel")
        summary.pack(anchor=tk.W, pady=(0, 4))
        note = ttk.Label(main, textvariable=self.note_var, style="Body.TLabel")
        note.pack(anchor=tk.W, pady=(0, 16))

        self.canvas = tk.Canvas(main, height=185, background="#eef2f7", highlightthickness=1, highlightbackground="#d1d5db")
        self.canvas.pack(fill=tk.X, pady=(0, 16))

        columns = ("number", "source", "destination", "protocol", "port", "status", "payload")
        self.tree = ttk.Treeview(main, columns=columns, show="headings", height=10)
        headings = {
            "number": "No.",
            "source": "Source IP",
            "destination": "Destination IP",
            "protocol": "Protocol",
            "port": "Port",
            "status": "Status",
            "payload": "Payload / Description",
        }
        widths = {
            "number": 50,
            "source": 135,
            "destination": 150,
            "protocol": 90,
            "port": 70,
            "status": 110,
            "payload": 390,
        }
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], anchor=tk.W, stretch=column == "payload")
        self.tree.pack(fill=tk.BOTH, expand=True)

        explanation = ttk.Label(
            main,
            text="Viva tip: DNS uses port 53 to resolve a domain into an IP address. HTTPS commonly uses TCP port 443 after the TCP 3-way handshake.",
            style="Body.TLabel",
        )
        explanation.pack(anchor=tk.W, pady=(14, 0))

    def simulate(self) -> None:
        domain = self.domain_var.get().strip() or DEFAULT_DOMAIN
        server_ip, is_live, note = resolve_domain(domain)
        self.packets = build_packets(domain, server_ip)

        lookup_label = "Live DNS" if is_live else "Demo fallback"
        self.summary_var.set(
            f"Client {CLIENT_IP} resolves {domain} using DNS server {DNS_SERVER_IP}, then connects to {server_ip} on TCP port 443."
        )
        self.note_var.set(f"{lookup_label}: {note}")

        for item in self.tree.get_children():
            self.tree.delete(item)
        for packet in self.packets:
            self.tree.insert(
                "",
                tk.END,
                values=(
                    packet.number,
                    packet.source,
                    packet.destination,
                    packet.protocol,
                    packet.port,
                    packet.status,
                    packet.payload,
                ),
            )

        self._draw_diagram(domain, server_ip)

    def _draw_diagram(self, domain: str, server_ip: str) -> None:
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 900)
        y = 90
        nodes = [
            (120, "Client", CLIENT_IP),
            (width // 2, "DNS Server", DNS_SERVER_IP),
            (width - 120, "Web Server", server_ip),
        ]

        for x, title, subtitle in nodes:
            self.canvas.create_rectangle(x - 86, y - 42, x + 86, y + 42, fill="#ffffff", outline="#9ca3af", width=2)
            self.canvas.create_text(x, y - 10, text=title, fill="#111827", font=("Arial", 13, "bold"))
            self.canvas.create_text(x, y + 16, text=subtitle, fill="#374151", font=("Arial", 10))

        self._arrow(210, y - 25, width // 2 - 96, y - 25, "DNS query: " + domain)
        self._arrow(width // 2 + 96, y + 25, 210, y + 25, "DNS response: " + server_ip)
        self._arrow(width // 2 + 96, y - 25, width - 210, y - 25, "TCP SYN -> SYN-ACK -> ACK")
        self._arrow(width - 210, y + 25, width // 2 + 96, y + 25, "HTTPS ready on port 443")

    def _arrow(self, x1: int, y1: int, x2: int, y2: int, label: str) -> None:
        self.canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST, width=2, fill="#2563eb")
        self.canvas.create_text((x1 + x2) // 2, y1 - 12, text=label, fill="#1f2937", font=("Arial", 9))


class PacketJourneyRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(PROJECT_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/resolve":
            params = parse_qs(parsed.query)
            raw_domains = params.get("domains", [DEFAULT_DOMAIN])[0]
            results = [build_resolution_result(domain) for domain in split_domain_inputs(raw_domains)]
            self._send_json({"results": results})
            return

        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def run_web_server(port: int, open_browser: bool) -> int:
    server_address = ("127.0.0.1", port)
    with ThreadingHTTPServer(server_address, PacketJourneyRequestHandler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        print(f"Packet Journey Visualizer running at {url}")
        print("Press Ctrl+C to stop the server.")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize DNS resolution and TCP connection packets.")
    parser.add_argument("--cli", action="store_true", help="Run in terminal mode instead of opening the GUI.")
    parser.add_argument("--web", action="store_true", help="Run the browser UI with a local DNS API.")
    parser.add_argument("--port", type=int, default=8000, help="Port for --web mode.")
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically in --web mode.")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN, help="Domain name to resolve and simulate.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.cli:
        return render_cli(args.domain)
    if args.web:
        return run_web_server(args.port, not args.no_open)

    if tk is None:
        print("Tkinter is not installed for this Python, so GUI mode cannot start.")
        print("Run the browser UI with real DNS support instead:")
        print(f"  {sys.executable} app.py --web")
        print()
        print("Or run terminal mode:")
        print(f"  {sys.executable} app.py --cli --domain {args.domain}")
        print()
        print("Opening index.html directly still works, but it cannot perform live DNS by itself.")
        return 1

    root = tk.Tk()
    PacketJourneyApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
