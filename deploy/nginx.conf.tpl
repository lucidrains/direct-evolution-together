location = {{SUBPATH}} {
    return 301 $scheme://$http_host{{SUBPATH}}/;
}

location ^~ {{SUBPATH}}/ws/ {
    proxy_pass http://127.0.0.1:{{WS_PORT}}/ws/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $http_host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_read_timeout 86400;
}

location ^~ {{SUBPATH}}/ {
    proxy_pass http://127.0.0.1:{{HTTP_PORT}}/;
    proxy_http_version 1.1;
    proxy_set_header Host $http_host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-Prefix {{SUBPATH}};
    client_max_body_size 4m;
}
