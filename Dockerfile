FROM tensorflow/serving:2.15.1

ARG MODEL_NAME=heart
ENV MODEL_NAME=${MODEL_NAME}
ENV MODEL_DIR=/models/${MODEL_NAME}
ENV GRPC_PORT=8500
ENV MONITORING_CONFIG=/monitoring.config
ENV TF_CPP_MIN_LOG_LEVEL=1
ENV ENABLE_ONEDNN_OPTS=0

COPY monitoring/prometheus.config ${MONITORING_CONFIG}

COPY serving_model/arusdian/ /tmp/arusdian/
RUN set -e; \
    test -d /tmp/arusdian || (echo "serving_model/arusdian/ not found in build context" && exit 1); \
    chosen="$(ls -1 /tmp/arusdian | grep -E '^[0-9]+$' | sort -n | tail -n 1 || true)"; \
    if [ -z "$chosen" ]; then chosen="$(ls -1t /tmp/arusdian | head -n 1 || true)"; fi; \
    echo "Chosen model version: $chosen"; \
    test -n "$chosen"; \
    mkdir -p "${MODEL_DIR}/1"; \
    cp -R "/tmp/arusdian/${chosen}/"* "${MODEL_DIR}/1/"; \
    test -f "${MODEL_DIR}/1/saved_model.pb" || (echo "saved_model.pb not found" && exit 1)

EXPOSE 8501 8500

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=5 \
    CMD wget -qO- "http://127.0.0.1:${PORT:-8501}/v1/models/${MODEL_NAME}" | grep -q AVAILABLE || exit 1

CMD /bin/sh -c '/usr/bin/tf_serving_entrypoint.sh \
    --rest_api_port=${PORT:-8501} \
    --port=${GRPC_PORT:-8500} \
    --model_name=${MODEL_NAME} \
    --model_base_path=${MODEL_DIR} \
    --monitoring_config_file=${MONITORING_CONFIG}'
