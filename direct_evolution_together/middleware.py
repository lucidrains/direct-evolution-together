from django.middleware.gzip import GZipMiddleware


_SKIP_GZIP_PREFIXES = (
    "video/",
    "audio/",
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "image/avif",
)
_SKIP_GZIP_EXACT = {
    "application/zip",
    "application/x-zip-compressed",
    "application/octet-stream",
    "application/gzip",
}


class MediaSafeGZipMiddleware(GZipMiddleware):
    """Avoid compressing already-compressed media/archives and preserve HTTP Range bytes."""

    def process_response(self, request, response):
        if response.status_code == 206 or response.has_header("Content-Range"):
            return response
        ctype = (response.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if ctype.startswith(_SKIP_GZIP_PREFIXES) or ctype in _SKIP_GZIP_EXACT:
            return response
        return super().process_response(request, response)
