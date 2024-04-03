# Use a slim Python image for efficiency (adjust base image if needed)
FROM continuumio/anaconda3

# Set working directory
WORKDIR /app

# Copy your application code and environment.yml file
COPY environment.yml /app/environment.yml
COPY boot.sh /app/boot.sh
RUN chmod +x boot.sh

RUN apt-get update && \
    apt-get install -y libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* \
RUN apt-get update && apt-get install libsm6 libxext6  -y
# Create a conda environment named 'myenv' (adjust name if needed)
RUN conda env create -f environment.yml
RUN echo "source activate AICam2" > ~/.bashrc
ENV PATH /opt/conda/envs/AICam2/bin:$PATH

# Install dependencies from environment.yml


# Install gunicorn (outside conda for compatibility)
# (Optional if your application uses a web framework)
RUN pip install gunicorn protobuf==4.25.3 numpy==1.23.5 keras==2.12.0 aiortc==1.8.0 tensorflow==2.12.0

COPY . /app/
# Expose port (adjust port number if needed)
EXPOSE 8080

# Command to start your Flask application using gunicorn (optional)
CMD ["gunicorn", "--bind", ":8080", "tensorflow1:app"]

# (Alternative command for non-web applications)
# CMD ["python", "your_main_script.py"]  # Replace with your main Python script