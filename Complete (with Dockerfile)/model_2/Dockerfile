# Python base image
FROM python:3.9-slim

# workdir
WORKDIR /app

# copy requirements.txt
COPY requirements.txt /app/requirements.txt

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy all the other files into the container
COPY . /app

# Flask app's port
EXPOSE 5000

# command to run the Flask application
CMD ["python", "app.py"]