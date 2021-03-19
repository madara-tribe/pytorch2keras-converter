IMAGE=torch2keras
run:
	docker build -t ${IMAGE} .
in:
	NV_GPU=0 nvidia-docker run --runtime=nvidia -v /home/hagi:/home/app/sig --rm -it ${IMAGE}:latest bash
