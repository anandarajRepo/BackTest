#!/usr/bin/env python3
"""
BackTest CLI - Fyers Authentication & Strategy Runner

Usage:
    python main.py auth          # Generate Fyers access token
    python main.py auth --port 8080  # Use custom port for redirect server
"""

import argparse
import os
import sys
import threading
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv, set_key

load_dotenv()


class AuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures the auth code from Fyers redirect."""

    auth_code = None

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if "auth_code" in params:
            AuthCallbackHandler.auth_code = params["auth_code"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h2>Authentication successful!</h2>"
                b"<p>Auth code captured. You can close this tab and return to the terminal.</p>"
                b"</body></html>"
            )
        elif "error" in params:
            error_msg = params.get("error", ["unknown"])[0]
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                f"<html><body><h2>Authentication failed</h2>"
                f"<p>Error: {error_msg}</p></body></html>".encode()
            )
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><p>Waiting for authentication...</p></body></html>"
            )

    def log_message(self, format, *args):
        """Suppress default HTTP server logs."""
        pass


def _validate_auth_credentials():
    """Check that required env vars are set for auth flow."""
    client_id = os.environ.get("FYERS_CLIENT_ID", "").strip()
    secret_key = os.environ.get("FYERS_SECRET_KEY", "").strip()
    redirect_uri = os.environ.get("FYERS_REDIRECT_URI", "").strip()

    missing = []
    if not client_id:
        missing.append("FYERS_CLIENT_ID")
    if not secret_key:
        missing.append("FYERS_SECRET_KEY")

    if missing:
        print(f"\nMissing required credentials: {', '.join(missing)}")
        print("Set them in your .env file. See .env.example for reference.")
        print("Get credentials from: https://myapi.fyers.in/dashboard/")
        sys.exit(1)

    if not redirect_uri:
        redirect_uri = "http://127.0.0.1:8000/"

    return client_id, secret_key, redirect_uri


def _extract_port_from_uri(redirect_uri):
    """Extract port number from the redirect URI."""
    parsed = urlparse(redirect_uri)
    if parsed.port:
        return parsed.port
    return 8000


def run_auth(port=None):
    """Run the Fyers OAuth2 authentication flow.

    1. Start a local HTTP server to catch the redirect
    2. Open browser to Fyers login page
    3. Capture the auth code from redirect
    4. Exchange auth code for access token
    5. Save access token to .env
    """
    try:
        from fyers_apiv3 import fyersModel
    except ImportError:
        print("\nfyers-apiv3 is not installed.")
        print("Install it with: pip install fyers-apiv3")
        sys.exit(1)

    client_id, secret_key, redirect_uri = _validate_auth_credentials()

    # Override port in redirect URI if --port was specified
    if port is not None:
        parsed = urlparse(redirect_uri)
        redirect_uri = f"{parsed.scheme}://127.0.0.1:{port}/"
    else:
        port = _extract_port_from_uri(redirect_uri)

    # Create session model
    session = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code",
    )

    # Generate auth URL
    auth_url = session.generate_authcode()

    if not auth_url or not isinstance(auth_url, str) or not auth_url.startswith("http"):
        print(f"\nFailed to generate auth URL. Response: {auth_url}")
        print("Check that your FYERS_CLIENT_ID and FYERS_SECRET_KEY are correct.")
        sys.exit(1)

    # Start local HTTP server to capture the redirect
    AuthCallbackHandler.auth_code = None
    server = HTTPServer(("127.0.0.1", port), AuthCallbackHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print(f"\nStarting Fyers authentication...")
    print(f"Redirect server listening on http://127.0.0.1:{port}/")
    print(f"\nOpening browser for login...")
    print(f"If the browser doesn't open, visit this URL manually:\n{auth_url}\n")

    webbrowser.open(auth_url)

    # Wait for the auth code (timeout after 120 seconds)
    timeout = 120
    start = time.time()
    while AuthCallbackHandler.auth_code is None:
        if time.time() - start > timeout:
            server.shutdown()
            print(f"\nTimeout: No auth code received within {timeout} seconds.")
            print("Try running the command again.")
            sys.exit(1)
        time.sleep(0.5)

    auth_code = AuthCallbackHandler.auth_code
    server.shutdown()

    print(f"Auth code received. Generating access token...")

    # Exchange auth code for access token
    session.set_token(auth_code)
    response = session.generate_token()

    if not isinstance(response, dict):
        print(f"\nUnexpected response from Fyers: {response}")
        sys.exit(1)

    if response.get("s") == "ok" and "access_token" in response:
        access_token = response["access_token"]

        # Save to .env file
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

        if not os.path.exists(env_path):
            # Create .env from .env.example if it doesn't exist
            example_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.example")
            if os.path.exists(example_path):
                with open(example_path, "r") as src, open(env_path, "w") as dst:
                    dst.write(src.read())
            else:
                with open(env_path, "w") as f:
                    f.write("")

        set_key(env_path, "FYERS_ACCESS_TOKEN", access_token)
        print(f"\nAccess token generated and saved to .env")
        print(f"Token preview: {access_token[:20]}...{access_token[-10:]}")

        # Verify the token works
        _verify_token(client_id, access_token)
    else:
        code = response.get("code", "unknown")
        message = response.get("message", response)
        print(f"\nFailed to generate access token.")
        print(f"Error code: {code}")
        print(f"Message: {message}")
        sys.exit(1)


def _verify_token(client_id, access_token):
    """Quick verification that the generated token works."""
    try:
        from fyers_apiv3 import fyersModel

        fyers = fyersModel.FyersModel(
            client_id=client_id,
            token=access_token,
            is_async=False,
            log_path="",
        )
        profile = fyers.get_profile()
        if profile.get("s") == "ok":
            name = profile.get("data", {}).get("name", "Unknown")
            print(f"Verified: Logged in as {name}")
        else:
            print("Warning: Token saved but profile verification failed.")
            print(f"Response: {profile}")
    except Exception as e:
        print(f"Warning: Token saved but verification failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="BackTest CLI - Trading Strategy Backtester & Fyers Auth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py auth              Generate Fyers access token
  python main.py auth --port 8080  Use custom port for redirect server
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # auth command
    auth_parser = subparsers.add_parser("auth", help="Generate Fyers access token via browser login")
    auth_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for local redirect server (default: from FYERS_REDIRECT_URI or 8000)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "auth":
        run_auth(port=args.port)


if __name__ == "__main__":
    main()
