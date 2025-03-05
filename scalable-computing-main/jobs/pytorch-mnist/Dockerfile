FROM python:3.12.0

ADD requirements.txt /opt/mnist/requirements.txt
RUN pip install -r /opt/mnist/requirements.txt
RUN mkdir -p /opt/mnist

WORKDIR /opt/mnist/src
ADD mnist.py /opt/mnist/src/mnist.py

RUN chgrp -R 0 /opt/mnist \
    && chmod -R g+rwX /opt/mnist

ENTRYPOINT ["python", "/opt/mnist/src/mnist.py"]