
DOCKER_CMD := docker run -it --rm --gpus=all --privileged=true --ipc=host -v $(shell pwd):/app
DOCKER_PY_CMD := ${DOCKER_CMD} --entrypoint=python


build-container:
	docker build -f docker/Dockerfile -t pytorch-video-pipeline:latest .


run-container: build-container
	${DOCKER_CMD} pytorch-video-pipeline:latest


logs/%.pipeline.dot: %.py
	${DOCKER_PY_CMD} pytorch-video-pipeline:latest $<


logs/%.pipeline.png: logs/%.pipeline.dot
	dot -Tpng -o$@ $< && rm -f $<


all: logs/frames_into_python.pipeline.png logs/frames_into_pytorch.pipeline.png

