

container:
	# create/get docker container
	docker build -f docker/Dockerfile -t pytorch-video-pipeline:latest .


run-container: container
	docker run -it --rm \
		--gpus=all \
		--privileged=true \
		--ipc=host \
		-v $(shell pwd):/app \
		pytorch-video-pipeline:latest


run-simple-cli: container
	# run simple pipeline from CLI using gst-launch-1.0
	docker run -it --rm \
		--gpus=all \
		--privileged=true \
		--ipc=host \
		-v $(shell pwd):/app \
		pytorch-video-pipeline:latest \
		gst-launch-1.0 filesrc location=media/in.mp4 ! decodebin ! progressreport update-freq=1 ! fakesink sync=true


run-simple: container
	# run simple.py in container
	docker run -it --rm \
		--gpus=all \
		--privileged=true \
		--ipc=host \
		-v $(shell pwd):/app \
		pytorch-video-pipeline:latest \
		python simple.py


