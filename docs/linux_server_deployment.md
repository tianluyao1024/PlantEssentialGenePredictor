# Linux server deployment

This guide targets Ubuntu 22.04 or 24.04 LTS. The recommended production
layout separates code/models from pretrained PLM weights:

```text
/opt/PlantEssentialGenePredictor/
  PlantEssentialGenePredictor/       GitHub code plus extracted Zenodo artifact
  plm_model_weights/                 ESM2, ProtBERT and ProtT5 weights
  raw_data/go-basic.obo              Optional GO hierarchy file
```

## 1. Transfer files

On the server, create the parent directory:

```bash
sudo mkdir -p /opt/PlantEssentialGenePredictor
sudo chown "$USER" /opt/PlantEssentialGenePredictor
cd /opt/PlantEssentialGenePredictor
git clone https://github.com/tianluyao1024/PlantEssentialGenePredictor.git
```

Download the Zenodo artifact (DOI `10.5281/zenodo.21387076`) and extract its
`models/`, `data/` and `predictions/` contents into:

```text
/opt/PlantEssentialGenePredictor/PlantEssentialGenePredictor/
```

Copy `plm_model_weights/` from the old machine to the parent directory. For
example, from the old machine:

```bash
rsync -avh --progress /path/to/plm_model_weights/ user@SERVER_IP:/tmp/plm_model_weights/
```

Then move it on the server:

```bash
sudo mv /tmp/plm_model_weights /opt/PlantEssentialGenePredictor/plm_model_weights
```

Copy `go-basic.obo` to:

```text
/opt/PlantEssentialGenePredictor/raw_data/go-basic.obo
```

## 2. Install and start the service

```bash
cd /opt/PlantEssentialGenePredictor/PlantEssentialGenePredictor
sudo bash deploy/linux/install_linux_server.sh
```

Check the service:

```bash
sudo systemctl status plant-essential-gene-predictor
curl -I http://127.0.0.1:8501
```

The script installs Python dependencies, creates a restricted service user and
starts Streamlit only on the local loopback interface. It does not expose port
8501 directly to the internet.

## 3. Configure domain and HTTPS

At the domain registrar, create DNS A records for `@` and `www` pointing to
the server public IPv4 address. Wait until DNS resolves, then run:

```bash
sudo cp deploy/linux/plantessentialgene.com.nginx.conf /etc/nginx/sites-available/plantessentialgene.com
sudo ln -sf /etc/nginx/sites-available/plantessentialgene.com /etc/nginx/sites-enabled/plantessentialgene.com
sudo nginx -t
sudo systemctl reload nginx
```

Install a certificate after DNS is active:

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d plantessentialgene.com -d www.plantessentialgene.com
```

Allow only SSH, HTTP and HTTPS at the cloud security-group/firewall layer.
Do not open public port 8501.

## 4. GPU verification

For raw FASTA jobs, verify the NVIDIA driver and PyTorch CUDA support:

```bash
nvidia-smi
/opt/PlantEssentialGenePredictor/PlantEssentialGenePredictor/.venv/bin/python -c "import torch; print(torch.cuda.is_available())"
```

If PyTorch reports `False`, processed `.npz` predictions remain available, but
new-species PLM extraction will run much more slowly on CPU.

## Operations

```bash
sudo systemctl restart plant-essential-gene-predictor
sudo journalctl -u plant-essential-gene-predictor -f
cd /opt/PlantEssentialGenePredictor/PlantEssentialGenePredictor
sudo -u plantessential .venv/bin/python scripts/webapp/cleanup_jobs.py --max-age-hours 24
```

For a server physically located in mainland China, confirm the hosting
provider's current ICP filing requirements before exposing the public domain.
