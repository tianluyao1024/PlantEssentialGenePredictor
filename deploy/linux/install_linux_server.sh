#!/usr/bin/env bash
set -euo pipefail

# Run after placing code plus the Zenodo artifact under /opt/PlantEssentialGenePredictor.
# Usage: sudo bash deploy/linux/install_linux_server.sh

APP_ROOT="${APP_ROOT:-/opt/PlantEssentialGenePredictor}"
PROJECT_ROOT="${PROJECT_ROOT:-${APP_ROOT}/PlantEssentialGenePredictor}"
APP_USER="${APP_USER:-plantessential}"
APP_PORT="${APP_PORT:-8501}"
SERVICE_NAME="plant-essential-gene-predictor"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this installer with sudo or as root."
  exit 1
fi

if [[ ! -f "${PROJECT_ROOT}/requirements.txt" ]]; then
  echo "Project code was not found at ${PROJECT_ROOT}."
  echo "Expected ${PROJECT_ROOT}/requirements.txt."
  exit 1
fi

required_paths=(
  "${PROJECT_ROOT}/models"
  "${PROJECT_ROOT}/data/processed_features"
  "${PROJECT_ROOT}/data/labels"
  "${PROJECT_ROOT}/predictions"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required Zenodo artifact path: ${path}"
    exit 1
  fi
done

apt-get update
apt-get install -y python3 python3-venv python3-pip nginx libgomp1 curl

if ! id "${APP_USER}" >/dev/null 2>&1; then
  useradd --system --create-home --home-dir "${APP_ROOT}" --shell /usr/sbin/nologin "${APP_USER}"
fi

python3 -m venv "${PROJECT_ROOT}/.venv"
"${PROJECT_ROOT}/.venv/bin/pip" install --upgrade pip
"${PROJECT_ROOT}/.venv/bin/pip" install -r "${PROJECT_ROOT}/requirements.txt"

chown -R "${APP_USER}:${APP_USER}" "${APP_ROOT}"

install -m 0644 "${PROJECT_ROOT}/deploy/linux/plant-essential-gene-predictor.service" \
  "/etc/systemd/system/${SERVICE_NAME}.service"
sed -i \
  -e "s|__APP_ROOT__|${APP_ROOT}|g" \
  -e "s|__PROJECT_ROOT__|${PROJECT_ROOT}|g" \
  -e "s|__APP_USER__|${APP_USER}|g" \
  -e "s|__APP_PORT__|${APP_PORT}|g" \
  "/etc/systemd/system/${SERVICE_NAME}.service"

systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}"
systemctl status "${SERVICE_NAME}" --no-pager

echo
echo "Application service is running on 127.0.0.1:${APP_PORT}."
echo "Next, install deploy/linux/plantessentialgene.com.nginx.conf in /etc/nginx/sites-available/ and configure DNS."
echo "Optional raw FASTA prediction assets:"
echo "  ${APP_ROOT}/plm_model_weights"
echo "  ${APP_ROOT}/raw_data/go-basic.obo"
