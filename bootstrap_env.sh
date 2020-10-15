pip install -r requirements.txt
rm -rf ~/.marlin/config
mkdir ~/.marlin
echo "[default]" > ~/.marlin/config
echo "client_id: $(date +%s)_" >> ~/.marlin/config

echo "------------- Bootstrap successful -------------"
