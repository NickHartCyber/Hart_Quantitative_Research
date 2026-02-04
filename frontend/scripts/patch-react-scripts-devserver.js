const fs = require('fs');
const path = require('path');

const targetPath = path.join(
  __dirname,
  '..',
  'node_modules',
  'react-scripts',
  'config',
  'webpackDevServer.config.js'
);

const logPrefix = '[patch-react-scripts]';

if (!fs.existsSync(targetPath)) {
  console.warn(`${logPrefix} ${targetPath} not found, skipping.`);
  process.exit(0);
}

const source = fs.readFileSync(targetPath, 'utf8');

if (source.includes('setupMiddlewares(middlewares, devServer)')) {
  console.log(`${logPrefix} setupMiddlewares already present, skipping.`);
  process.exit(0);
}

const beforeAfterRegex =
  / {4}onBeforeSetupMiddleware\(devServer\)\s*\{[\s\S]*?\n {4}\},\n {4}onAfterSetupMiddleware\(devServer\)\s*\{[\s\S]*?\n {4}\},/;

const replacementLines = [
  "    setupMiddlewares(middlewares, devServer) {",
  "      // Keep `evalSourceMapMiddleware`",
  "      // middlewares before `redirectServedPath` otherwise will not have any effect",
  "      // This lets us fetch source contents from webpack for the error overlay",
  "      middlewares.unshift({",
  "        name: 'eval-source-map',",
  "        middleware: evalSourceMapMiddleware(devServer),",
  "      });",
  "",
  "      if (fs.existsSync(paths.proxySetup)) {",
  "        // This registers user provided middleware for proxy reasons",
  "        require(paths.proxySetup)(devServer.app);",
  "      }",
  "",
  "      // Redirect to `PUBLIC_URL` or `homepage` from `package.json` if url not match",
  "      middlewares.push({",
  "        name: 'redirect-served-path',",
  "        middleware: redirectServedPath(paths.publicUrlOrPath),",
  "      });",
  "",
  "      // This service worker file is effectively a 'no-op' that will reset any",
  "      // previous service worker registered for the same host:port combination.",
  "      // We do this in development to avoid hitting the production cache if",
  "      // it used the same host and port.",
  "      // https://github.com/facebook/create-react-app/issues/2272#issuecomment-302832432",
  "      middlewares.push({",
  "        name: 'noop-service-worker',",
  "        middleware: noopServiceWorkerMiddleware(paths.publicUrlOrPath),",
  "      });",
  "",
  "      return middlewares;",
  "    },",
];

const replacement = replacementLines.join('\n');
const updated = source.replace(beforeAfterRegex, replacement);

if (updated === source) {
  console.warn(`${logPrefix} Unable to patch webpackDevServer config, skipping.`);
  process.exit(0);
}

fs.writeFileSync(targetPath, updated, 'utf8');
console.log(`${logPrefix} Patched webpackDevServer config to use setupMiddlewares.`);
