FROM python:3.13-slim-bookworm AS imagej

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl openjdk-17-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

RUN set -eu; \
    download() { curl --fail --show-error --silent --location --retry 2 --retry-all-errors \
        --connect-timeout 15 --max-time 60 --output "$1" "$2"; }; \
    mkdir -p /opt/glcm-imagej/plugins; \
    download /opt/glcm-imagej/ij.jar https://sites.imagej.net/Fiji/jars/ij-1.54p.jar-20250219110715 \
      || download /opt/glcm-imagej/ij.jar https://repo.maven.apache.org/maven2/net/imagej/ij/1.54p/ij-1.54p.jar; \
    download /opt/glcm-imagej/plugins/GLCM_Texture.class https://wsr.imagej.net/ij/plugins/download/GLCM_Texture.class \
      || download /opt/glcm-imagej/plugins/GLCM_Texture.class https://imagej.net/ij/plugins/download/GLCM_Texture.class

COPY docker/GlcmMacroRunner.java /tmp/
RUN javac -cp /opt/glcm-imagej/ij.jar -d /opt/glcm-imagej /tmp/GlcmMacroRunner.java

FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GLCM_FIJI_EXE=/usr/local/bin/glcm-fiji \
    GLCM_FIJI_MODE=batch

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-dejavu-core openjdk-17-jre xauth xvfb \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 texture

COPY --from=imagej /opt/glcm-imagej/ /opt/glcm-imagej/
COPY --chmod=755 docker/glcm-fiji /usr/local/bin/glcm-fiji

WORKDIR /app
COPY requirements.txt requirements-web.txt ./
RUN python -m pip install --no-cache-dir -r requirements-web.txt

COPY web_app.py evaluate_glcm_models.py glcm.py glcm_classification_results.json ./
COPY glcm_G16_d1_a0.csv glcm_L16_d1_a0_1794925531103.csv GLCM_Texture_Batch_L16_d1_a0.ijm ./
COPY web/ web/
COPY dataset_xla/ dataset_xla/
COPY docker/smoke.py docker/

USER texture
RUN python -m docker.smoke

EXPOSE 8000
CMD ["python", "web_app.py", "--host", "0.0.0.0", "--port", "8000"]
