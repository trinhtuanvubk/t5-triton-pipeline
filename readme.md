# T5 Triton

docker run --gpus=1 -itd --add-host=host.docker.internal:host-gateway -p 8050-8052:8000-8002 -v ${PWD}/model_repository:/models --shm-size 2gb --name jbtest tttest:latest tritonserver --model-repository=/models
