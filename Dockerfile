# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set work directory
WORKDIR /app

# Install pipenv
RUN pip install --upgrade pip
RUN pip install pipenv

# Copy Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Install dependencies
RUN pipenv install --system --deploy

# Copy the current directory contents into the container at /app
COPY . .

# Make port 9696 available to the world outside this container
EXPOSE 9696

# Run predict.py when the container launches
CMD ["python", "predict.py"]