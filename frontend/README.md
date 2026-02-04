# Hart Quantitative Research frontend

## Run in Docker (uses npm inside the container)
- From the repo root: `docker compose up frontend --build`  
  or start the whole stack: `docker compose up --build`
- Dev server: http://localhost:3000 (proxied API base: http://localhost:8000 via `REACT_APP_API_BASE`).
- Live reload works via bind mounts; rebuild the image after dependency changes: `docker compose build frontend`.

## Run locally (without Docker)
```
npm install
npm start
```

## Scripts
- `npm start` — CRA dev server
- `npm test` — test runner
- `npm run build` — production build to `build/`
