# Frontend container to run the CRA dev server inside Docker
FROM node:20-alpine

WORKDIR /app

ENV PATH="/app/node_modules/.bin:${PATH}" \
    HOST=0.0.0.0 \
    PORT=3000 \
    CHOKIDAR_USEPOLLING=true

COPY package.json package-lock.json ./
COPY scripts ./scripts
RUN npm ci

COPY public ./public
COPY src ./src

EXPOSE 3000

CMD ["npm", "start"]
