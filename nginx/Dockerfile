FROM nginx

# Remove the default nginx.conf
RUN rm /etc/nginx/conf.d/default.conf

#COPY nginx.conf /etc/nginx/nginx.conf
COPY conf.d/app.conf /etc/nginx/conf.d/app.conf

EXPOSE 80 443
CMD ["nginx", "-g", "daemon off;"]