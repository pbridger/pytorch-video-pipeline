
DOCKER_CMD := docker run -it --rm --gpus=all --privileged=true --ipc=host -v $(shell pwd):/app
DOCKER_PY_CMD := ${DOCKER_CMD} --entrypoint=python

.PHONY: sleep


build-container: docker/Dockerfile
	docker build -f $< -t pytorch-video-pipeline:latest .


run-container: build-container
	${DOCKER_CMD} pytorch-video-pipeline:latest


logs/cli.pipeline.dot:
	${DOCKER_CMD} --entrypoint=gst-launch-1.0 pytorch-video-pipeline:latest filesrc location=media/in.mp4 num-buffers=200 ! decodebin ! progressreport update-freq=1 ! fakesink sync=true


logs/%.pipeline.dot: %.py
	${DOCKER_PY_CMD} pytorch-video-pipeline:latest $<


%.pipeline.png: logs/%.pipeline.dot
	dot -Tpng -o$@ $< && rm -f $<


%.output.svg: %.rec
	cat $< | svg-term > $@
	
%.rec:
	asciinema rec $@ -c "$(MAKE) --no-print-directory logs/$*.pipeline.dot sleep"

sleep:
	@sleep 2
	@echo "---"

all: cli.pipeline.png frames_into_python.pipeline.png frames_into_pytorch.pipeline.png

