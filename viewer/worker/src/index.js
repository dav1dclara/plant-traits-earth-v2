// Serves objects from a private R2 bucket with HTTP Range + CORS support, so the
// PMTiles client can fetch byte ranges directly. The bucket is never public; all
// access goes through this Worker (bound as BUCKET in wrangler.toml).
//
// GET https://<worker>.workers.dev/dem.pmtiles  ->  byte ranges of that object.

const CORS = {
  "access-control-allow-origin": "*", // tighten to your viewer origin(s) if desired
  "access-control-allow-methods": "GET, HEAD, OPTIONS",
  "access-control-allow-headers": "range, if-match, if-none-match",
  "access-control-expose-headers": "content-length, content-range, accept-ranges, etag",
  "access-control-max-age": "86400",
};

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: CORS });
    }
    if (request.method !== "GET" && request.method !== "HEAD") {
      return new Response("Method Not Allowed", { status: 405, headers: CORS });
    }

    const key = decodeURIComponent(new URL(request.url).pathname.slice(1));
    if (!key) return new Response("Not Found", { status: 404, headers: CORS });

    // Passing the request headers lets R2 parse the Range header for us.
    const object = await env.BUCKET.get(key, { range: request.headers });
    if (!object) return new Response("Not Found", { status: 404, headers: CORS });

    const headers = new Headers(CORS);
    object.writeHttpMetadata(headers); // content-type etc. from stored metadata
    headers.set("etag", object.httpEtag);
    headers.set("accept-ranges", "bytes");
    headers.set("cache-control", "public, max-age=3600");

    let status = 200;
    if (object.range) {
      let offset, length;
      if ("suffix" in object.range && object.range.suffix != null) {
        length = object.range.suffix;
        offset = object.size - length;
      } else {
        offset = object.range.offset ?? 0;
        length = object.range.length ?? object.size - offset;
      }
      headers.set("content-range", `bytes ${offset}-${offset + length - 1}/${object.size}`);
      headers.set("content-length", String(length));
      status = 206;
    } else {
      headers.set("content-length", String(object.size));
    }

    return new Response(request.method === "HEAD" ? null : object.body, { status, headers });
  },
};
