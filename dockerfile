# TensorFlow Serving 2.15
FROM tensorflow/serving:2.15.1

# Monitoring untuk Prometheus
COPY monitoring/prometheus.config /monitoring.config

# ==== ENV default ====
ARG MODEL_NAME=heart
ENV MODEL_NAME=${MODEL_NAME}
ENV TF_PORT=8501
ENV GRPC_PORT=8500
ENV MONITORING_CONFIG=/monitoring.config

# Direktori model final di dalam container (tanpa dobel)
ENV MODEL_DIR=/models/${MODEL_NAME}

# Salin versi terbaru ke /models/<MODEL_NAME>/1
COPY serving_model/arusdian/ /tmp/arusdian/
RUN set -e; \
    test -d /tmp/arusdian || (echo "serving_model/arusdian/ not found in build context" && exit 1); \
    echo "Available versions:" && ls -1 /tmp/arusdian || true; \
    latest_numeric=$(ls -1 /tmp/arusdian | grep -E '^[0-9]+$' | sort -n | tail -n 1 || true); \
    if [ -n "$latest_numeric" ]; then chosen="$latest_numeric"; else chosen=$(ls -1t /tmp/arusdian | head -n 1 || true); fi; \
    echo "Chosen model version: $chosen"; \
    test -n "$chosen"; \
    mkdir -p ${MODEL_DIR}/1; \
    cp -R "/tmp/arusdian/$chosen/"* ${MODEL_DIR}/1/; \
    test -f ${MODEL_DIR}/1/saved_model.pb || (echo "saved_model.pb not found" && exit 1)

EXPOSE ${TF_PORT}
EXPOSE ${GRPC_PORT}

# Pakai shell form agar ENV diexpand
CMD /bin/sh -c "/usr/bin/tf_serving_entrypoint.sh \
    --rest_api_port=${TF_PORT} \
    --port=${GRPC_PORT} \
    --model_name=${MODEL_NAME} \
    --model_base_path=${MODEL_DIR} \
    --monitoring_config_file=${MONITORING_CONFIG}"
