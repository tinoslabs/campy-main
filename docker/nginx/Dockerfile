ARG DOMAIN

FROM nginx:stable-alpine

ENV DOMAIN=${DOMAIN}

# Copy custom Nginx configuration
COPY ./docker/nginx/includes /etc/nginx/includes
COPY ./docker/nginx/nginx.conf /etc/nginx/nginx.conf

# Set the working directory to the public folder
WORKDIR /var/www/

COPY ./docker/nginx/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["entrypoint.sh"]

CMD ["nginx", "-g", "daemon off;"]
