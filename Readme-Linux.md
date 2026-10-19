python3 -m venv .venv
source .venv/bin/activate
pwsh ./run_phase1.ps1

python ./src/train_with_mlflow.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
python -m pip freeze > requirements-lock.txt

git config --global user.name "Santosh Srinivas G"
git config --global user.email "2025ae05691@wilp.bits-pilani.ac.in"

docker:

sudo dnf -y install dnf-plugins-core
sudo dnf config-manager --add-repo https://download.docker.com/linux/rhel/docker-ce.repo
sudo dnf install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo docker run hello-world

sudo docker run --rm --name heart-disease-api -p 8000:8000 heart-disease-api:2.0

# Install kubectl and Minikube on Rocky Linux:

curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
rm kubectl

curl -LO https://github.com/kubernetes/minikube/releases/latest/download/mi
nikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
rm minikube-linux-amd64

# Minikube needs permission to use Docker. You can grant your account access, then start a new shell with the updated group:

sudo usermod -aG docker "$USER"
newgrp docker

#  Start the cluster
minikube start --driver=docker
kubectl get nodes

# Push images to docker hub
docker login -u santoshsrinivasg
docker tag heart-disease-api:2.0 santoshsrinivasg/heart-disease-api:2.0
docker push santoshsrinivasg/heart-disease-api:2.0


kubectl apply -f k8s/deployment.yml -f k8s/service.yaml
kubectl rollout status deployment/heart-disease-api
kubectl get pods,services