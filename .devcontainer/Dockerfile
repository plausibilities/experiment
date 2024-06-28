# Starting with ...
FROM nvcr.io/nvidia/jax:23.08-py3


# If the steps of a `Dockerfile` use files that are different from the `context` file, COPY the
# file of each step separately; and RUN the file immediately after COPY
WORKDIR /app
COPY /.devcontainer/requirements.txt /app

# pip
RUN pip install --upgrade pip && pip install -r requirements.txt --no-cache

# Port
EXPOSE 8050

# ENTRYPOINT
ENTRYPOINT ["bash"]
