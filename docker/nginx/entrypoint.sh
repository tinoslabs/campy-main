#!/bin/sh

SSL_CERT_PATH="/etc/letsencrypt/live/${DOMAIN}/fullchain.pem"
SSL_KEY_PATH="/etc/letsencrypt/live/${DOMAIN}/privkey.pem"

if [ -f "$SSL_CERT_PATH" ] && [ -f "$SSL_KEY_PATH" ]; then
    echo "SSL certificate found. Starting Nginx..."
    sed -i "s/__DOMAIN__/$DOMAIN/g" /etc/nginx/includes/ssl.conf
    cp  /etc/nginx/includes/ssl.conf /etc/nginx/includes/server.conf
else
    echo "SSL certificate not found. Starting Nginx in non-SSL mode..."
    cp /etc/nginx/includes/init.conf /etc/nginx/includes/server.conf
fi

sh /docker-entrypoint.sh

exec "$@" 