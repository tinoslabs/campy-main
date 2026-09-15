#!/bin/sh

echo 'Running migrations...'
python manage.py migrate --no-input

echo 'Collecting static files...'
python manage.py collectstatic --no-input

exec gunicorn $PROJECT_NAME.wsgi:application --name $PROJECT_NAME\
    --workers 2 --threads 4 --worker-class gthread\
    --bind 0.0.0.0:8000 --log-level info
