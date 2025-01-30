FROM nvcr.io/nvidia/tritonserver:23.02-py3

# Install dependencies
RUN pip install opencv-python && \
    apt update && \
    apt install -y libgl1 && \
    rm -rf /var/lib/apt/lists/*

CMD ["tritonserver", "--grpc-port=8004", "--http-port=8005", "--metrics-port=8007","--model-repository=/models"]