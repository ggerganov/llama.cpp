: # This script will help setup a cloudflared tunnel for accessing KoboldCpp over the internet
: # It should work out of the box on both linux and windows
: # ======
: # WINDOWS PORTION
:<<BATCH
    @echo off
    echo Starting Cloudflare Tunnel for Windows
    curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe -o cloudflared.exe
    cloudflared.exe tunnel --url localhost:5001
    GOTO ENDING
BATCH
: # LINUX PORTION
echo 'Starting Cloudflare Tunnel for Linux'
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o 'cloudflared-linux-amd64' # 
chmod +x 'cloudflared-linux-amd64' # 
./cloudflared-linux-amd64 tunnel --url http://localhost:5001 # 
exit # 
:ENDING
