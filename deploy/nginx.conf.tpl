{{SUBPATH_REDIRECT_BLOCK}}location ^~ {{SUBPATH}}/ws/ {
    proxy_pass http://127.0.0.1:{{WS_PORT}}/ws/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $http_host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_buffering off;
    proxy_read_timeout 86400;
    proxy_send_timeout 86400;
}

location ^~ {{SUBPATH}}/media/ {
    alias {{MEDIA_DIR}}/;
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    gzip off;
    expires 365d;
    add_header Cache-Control "public, max-age=31536000, immutable" always;
    add_header X-Content-Type-Options "nosniff" always;
    access_log off;
}

location {{HTTP_LOC_MOD}}{{SUBPATH}}/ {
    proxy_pass http://127.0.0.1:{{HTTP_PORT}}/;
    proxy_http_version 1.1;
    proxy_set_header Host $http_host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
{{FORWARDED_PREFIX_LINE}}    client_max_body_size 4m;

    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_min_length 256;
    gzip_buffers 16 8k;
    gzip_types
        text/plain
        text/css
        text/javascript
        text/x-python
        text/xml
        application/json
        application/javascript
        application/x-javascript
        application/manifest+json
        application/xml
        application/wasm
        image/svg+xml
        image/x-icon;

    proxy_buffering on;
    proxy_buffer_size 16k;
    proxy_buffers 16 16k;
    proxy_busy_buffers_size 32k;
}
