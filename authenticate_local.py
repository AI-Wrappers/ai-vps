import os
import json
import base64
from google_auth_oauthlib.flow import Flow

def main():
    secret_b64 = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET_BASE64")
    if not secret_b64:
        print("Error: GOOGLE_OAUTH_CLIENT_SECRET_BASE64 environment variable not found.")
        print("Please export it as a base64 encoded string of your client_secret.json desktop credentials.")
        return

    try:
        client_config = json.loads(base64.b64decode(secret_b64).decode("utf-8"))
    except Exception as e:
        print(f"Error decoding GOOGLE_OAUTH_CLIENT_SECRET_BASE64: {e}")
        return

    # Using drive.file scope so the token only has access to files/folders created by this app
    flow = Flow.from_client_config(
        client_config,
        scopes=["https://www.googleapis.com/auth/drive.file"],
        redirect_uri="http://localhost"
    )
    auth_url, _ = flow.authorization_url(prompt="consent")

    print("\n=== LOCAL GOOGLE DRIVE AUTHENTICATION ===")
    print("1. Open the following link in your browser:")
    print(auth_url)
    print("2. Log in and authorize the application.")
    print("3. You will be redirected to a page that fails to load (e.g. http://localhost/?code=4/0Afu...)")
    print("4. Copy the ENTIRE redirect URL from your browser's address bar and paste it below.")

    redirect_url = input("\nEnter the full redirect URL (or authorization code): ").strip()

    if "code=" in redirect_url:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(redirect_url)
        code = parse_qs(parsed.query).get("code", [redirect_url])[0]
    else:
        code = redirect_url

    print("\nFetching token...")
    flow.fetch_token(code=code)
    creds = flow.credentials

    creds_json = creds.to_json()
    creds_b64 = base64.b64encode(creds_json.encode("utf-8")).decode("utf-8")

    print("\nSUCCESS!")
    print("Add this environment variable to your VPS:")
    print(f"\nexport GOOGLE_OAUTH_TOKEN_BASE64=\"{creds_b64}\"\n")

if __name__ == "__main__":
    main()
