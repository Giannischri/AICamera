# Use a slim Python image for efficiency (adjust base image if needed)
FROM python:3.9-slim
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
#ENV LD_LIBRARY_PATH="/your/path:${LD_LIBRARY_PATH}"

RUN apt-get update && apt -y install curl gnupg python3-venv libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy your application code and environment.yml file
#COPY environment.yml /app/environment.yml
#COPY boot.sh /app/boot.sh
COPY requirements.txt /app/requirements.txt
COPY . /app/
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install wheel
RUN pip install aiortc gunicorn flask-socketio flask pandas==2.2.1 numpy==1.23.5 opencv-python
RUN curl -O https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.15.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
RUN pip install tensorflow_cpu-2.15.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
#RUN pip install --upgrade pip
#RUN pip install wheel
#RUN pip install aiortc
#ENV LD_LIBRARY_PATH="$HOME/.conda/envs/glibcxx_test/lib/"
# Install dependencies from environment.yml



# Install gunicorn (outside conda for compatibility)
# (Optional if your application uses a web framework)
#RUN pip install gunicorn protobuf==4.25.3 numpy==1.23.5 keras==2.12.0 aiortc==1.8.0  pandas tensorflow==2.12.0
#RUN pip install opencv-python
#RUN pip install flask-socketio


# Expose port (adjust port number if needed)
EXPOSE 5000


# Command to start your Flask application using gunicorn (optional)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app1:app"]
#CMD ["python", "app1.py"]
# (Alternative command for non-web applications)