#!/usr/bin/env python3
"""
BackTest CLI - Fyers Authentication & Strategy Runner

Usage:
    python main.py auth    # Generate Fyers access token
"""

import argparse
import os
import sys

from dotenv import load_dotenv, set_key

load_dotenv()


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


def run_auth():
    """Run the Fyers OAuth2 authentication flow.

    1. Generate the auth URL
    2. User visits the URL and logs in
    3. User copies the auth code from the redirect URL
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

    # Step 1: Show the URL
    print("\n--- Fyers Authentication ---\n")
    print("Step 1: Open this URL in your browser and log in:\n")
    print(auth_url)
    print("\nStep 2: After login, you will be redirected.")
    print("         Copy the 'auth_code' value from the redirect URL.")
    print("         (It appears after ?auth_code= in the address bar)\n")

    # Step 2: User pastes the auth code
    auth_code = input("Step 3: Paste the auth code here: ").strip()

    if not auth_code:
        print("\nNo auth code provided. Exiting.")
        sys.exit(1)

    print("\nGenerating access token...")

    # Step 3: Exchange auth code for access token
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
  python main.py auth    Generate Fyers access token
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # auth command
    subparsers.add_parser("auth", help="Generate Fyers access token via browser login")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "auth":
        run_auth()


if __name__ == "__main__":
    main()
