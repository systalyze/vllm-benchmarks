FROM vllm/vllm-openai:v0.27.1
RUN pip install --no-cache-dir --no-deps "transformers==5.14.1" && pip check | grep -i transformers || true
