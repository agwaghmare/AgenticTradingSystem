#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq postgresql postgresql-contrib

sudo service postgresql start

sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname='agentictradingsystem'" | grep -q 1 \
  || sudo -u postgres psql -c "CREATE DATABASE agentictradingsystem;"

sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'localpassword';"

# Allow Windows host connections via localhost port forward
PG_CONF=$(sudo -u postgres psql -tAc "SHOW config_file;")
PG_HBA=$(sudo -u postgres psql -tAc "SHOW hba_file;")
sudo sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/" "$PG_CONF"
grep -q "0.0.0.0/0" "$PG_HBA" || echo "host all all 0.0.0.0/0 scram-sha-256" | sudo tee -a "$PG_HBA" > /dev/null

sudo service postgresql restart
echo "PostgreSQL ready in WSL"
