# TensorFlow Serving 2.15
FROM tensorflow/serving:2.15.1

COPY monitoring/prometheus.config /monitoring.config

ARG MODEL_NAME=heart
ENV MODEL_NAME=${MODEL_NAME}
ENV GRPC_PORT=8500
ENV MONITORING_CONFIG=/monitoring.config
ENV MODEL_DIR=/models/${MODEL_NAME}

COPY serving_model/arusdian/ /tmp/arusdian/
RUN set -e; \
    test -d /tmp/arusdian || (echo "serving_model/arusdian/ not found in build context" && exit 1); \
    latest_numeric=$(ls -1 /tmp/arusdian | grep -E '^[0-9]+$' | sort -n | tail -n 1 || true); \
    if [ -n "$latest_numeric" ]; then chosen="$latest_numeric"; else chosen=$(ls -1t /tmp/arusdian | head -n 1 || true); fi; \
    echo "Chosen model version: $chosen"; \
    mkdir -p ${MODEL_DIR}/1; \
    cp -R "/tmp/arusdian/$chosen/"* ${MODEL_DIR}/1/; \
    test -f ${MODEL_DIR}/1/saved_model.pb || (echo "saved_model.pb not found" && exit 1)

EXPOSE 8501
EXPOSE 8500

# REST pakai $PORT dari Railway, gRPC tetap 8500
CMD /bin/sh -c "/usr/bin/tf_serving_entrypoint.sh \
    --rest_api_port=${PORT:-8501} \
    --port=${GRPC_PORT:-8500} \
    --model_name=${MODEL_NAME} \
    --model_base_path=${MODEL_DIR} \
    --monitoring_config_file=${MONITORING_CONFIG}"
