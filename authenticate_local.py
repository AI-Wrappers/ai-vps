import os
import json
from google_auth_oauthlib.flow import Flow

def find_credentials_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.exists(os.path.join(current_dir, "_put_credentials_and_token_here")):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir
    return current_dir

def main():
    creds_dir = find_credentials_dir()
    credentials_path = os.path.join(creds_dir, "credentials.json")
    token_path = os.path.join(creds_dir, "token.json")

    if not os.path.exists(credentials_path):
        print(f"Error: Client credentials file not found at: {credentials_path}")
        print("Please download your OAuth client secret JSON (Desktop application) from Google Cloud Console,")
        print("place it there, and rename it to 'credentials.json'.")
        return

    try:
        with open(credentials_path, "r", encoding="utf-8") as f:
            client_config = json.load(f)
    except Exception as e:
        print(f"Error loading credentials from {credentials_path}: {e}")
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

    print(f"\nSaving token to {token_path}...")
    with open(token_path, "w", encoding="utf-8") as f:
        f.write(creds.to_json())

    print("\nSUCCESS!")
    print(f"Saved OAuth token to: {token_path}")

if __name__ == "__main__":
    main()
