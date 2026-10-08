#!/usr/bin/env bash
# Netlify build (see netlify.toml): switch the page to in-browser inference and,
# if API_BASE_URL is set, point the LLM note and docs link at the Modal API.
set -euo pipefail

sed -i 's#<meta name="inference" content="server">#<meta name="inference" content="browser">#' static/index.html
grep -q '<meta name="inference" content="browser">' static/index.html

if [ -n "${API_BASE_URL:-}" ]; then
  sed -i "s#<meta name=\"api-base\" content=\"\">#<meta name=\"api-base\" content=\"${API_BASE_URL%/}\">#" static/index.html
  grep -q "content=\"${API_BASE_URL%/}\"" static/index.html
fi

echo "inference=browser api-base=${API_BASE_URL:-<none>}"
