# Fyers API Setup Guide

## Problem: "Could not authenticate the user"

This error occurs when the Fyers API credentials are missing or incorrect. You need **both** a Client ID and Access Token to authenticate.

## Quick Fix

### Step 1: Create a `.env` file

Create a file named `.env` in the project root directory:

```bash
cp .env.example .env
```

### Step 2: Add Your Credentials

Edit the `.env` file and add your Fyers credentials:

```env
FYERS_CLIENT_ID=your_client_id_here
FYERS_ACCESS_TOKEN=your_access_token_here
```

## How to Get Fyers API Credentials

### 1. Get Your Client ID (App ID)

1. Login to Fyers API Dashboard: https://myapi.fyers.in/dashboard/
2. Click on "My Apps" or "Create App"
3. Create a new app if you don't have one
4. Your **Client ID** (format: `XXXXXXXX-XXX`) will be displayed
5. Copy this Client ID

### 2. Generate Access Token

The access token requires authentication flow. You have two options:

#### Option A: Using Fyers API Authentication (Recommended)

1. Visit the Fyers authentication docs: https://myapi.fyers.in/docsv3/#tag/Authentication
2. Follow the 3-step OAuth flow:
   - Step 1: Generate auth code
   - Step 2: Exchange auth code for access token
   - Step 3: Use the access token

#### Option B: Quick Token Generation Script

Create a file `generate_fyers_token.py`:

```python
from fyers_apiv3 import fyersModel
import os
from dotenv import load_dotenv

load_dotenv()

client_id = os.environ.get('FYERS_CLIENT_ID')
secret_key = "YOUR_SECRET_KEY"  # Get from Fyers dashboard
redirect_uri = "https://127.0.0.1:8000/"  # Your redirect URI

session = fyersModel.SessionModel(
    client_id=client_id,
    secret_key=secret_key,
    redirect_uri=redirect_uri,
    response_type="code",
    grant_type="authorization_code"
)

# Generate auth code URL
print("\nStep 1: Visit this URL to authenticate:")
print(session.generate_authcode())

# After authentication, you'll be redirected with an auth code
auth_code = input("\nStep 2: Enter the auth code from URL: ")

# Generate access token
session.set_token(auth_code)
response = session.generate_token()

if response['s'] == 'ok':
    print("\n✅ Access Token Generated Successfully!")
    print(f"Access Token: {response['access_token']}")
    print("\nAdd this to your .env file:")
    print(f"FYERS_ACCESS_TOKEN={response['access_token']}")
else:
    print(f"\n❌ Error: {response}")
```

Run: `python generate_fyers_token.py`

## Important Notes

### Token Expiry
- Access tokens expire daily (usually at market close)
- You need to regenerate the access token each trading day
- Consider implementing automatic token refresh in production

### Security
- Never commit `.env` file to git (already in .gitignore)
- Never share your Client ID, Secret Key, or Access Token
- Keep your credentials secure

### Testing Your Setup

After adding credentials to `.env`, test with:

```bash
python OpenRangeBreakout.py
```

If configured correctly, you should see:
```
Fetching data from Fyers for NSE:SBIN-EQ...
Loaded X candles from YYYY-MM-DD to YYYY-MM-DD
```

If you still see "Could not authenticate the user":
1. Verify Client ID format (should be like: `XXXXXXXX-XXX`)
2. Ensure Access Token is valid and not expired
3. Check that both credentials are in `.env` file
4. Verify `.env` file is in the same directory as the script

## Troubleshooting

### Error: "Could not authenticate the user"
- **Cause**: Missing or invalid Client ID or Access Token
- **Fix**: Verify both credentials are set correctly in `.env`

### Error: "Access token expired"
- **Cause**: Token expires daily
- **Fix**: Generate a new access token

### Error: "Invalid client_id"
- **Cause**: Incorrect Client ID format
- **Fix**: Copy the exact Client ID from Fyers dashboard

## References

- Fyers API Documentation: https://myapi.fyers.in/docsv3/
- Authentication Guide: https://myapi.fyers.in/docsv3/#tag/Authentication
- API Dashboard: https://myapi.fyers.in/dashboard/
- Python SDK: https://github.com/fyers-api/fyers-api-python
