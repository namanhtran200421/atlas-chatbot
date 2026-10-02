/** Start the Angular page and its local AWS signer, reusing either if already running. */
import { spawn } from 'node:child_process';

const WEB = 'http://localhost:4200';
const SIGNER = 'http://127.0.0.1:4300';
const children = [];

async function probe(url, expected) {
  try {
    const response = await fetch(url, { signal: AbortSignal.timeout(1000) });
    return response.ok && (await response.text()).includes(expected);
  } catch {
    return false;
  }
}

async function waitFor(url, expected, child) {
  for (let attempt = 0; attempt < 60; attempt += 1) {
    if (await probe(url, expected)) return;
    if (child.exitCode !== null) throw new Error(`${child.spawnargs.join(' ')} exited early`);
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`Timed out waiting for ${url}`);
}

function start(command, args, env = process.env) {
  const child = spawn(command, args, { stdio: 'inherit', env });
  children.push(child);
  return child;
}

function stop() {
  for (const child of children) child.kill('SIGTERM');
}

process.on('SIGINT', stop);
process.on('SIGTERM', stop);

try {
  if (!(await probe(`${SIGNER}/health`, 'membership-rag-signing-proxy'))) {
    const signer = start(process.execPath, ['tools/aws-proxy.mjs']);
    await waitFor(`${SIGNER}/health`, 'membership-rag-signing-proxy', signer);
  }

  if (!(await probe(WEB, 'Oriana'))) {
    // Invoke the CLI through Node instead of the platform-specific .bin shim.
    // Windows creates ng.cmd, while Unix creates ng, so spawning the shim path
    // directly fails with ENOENT on Windows.
    const web = start(
      process.execPath,
      ['node_modules/@angular/cli/bin/ng.js', 'serve'],
      { ...process.env, NG_CLI_ANALYTICS: process.env.NG_CLI_ANALYTICS ?? 'false' },
    );
    await waitFor(WEB, 'Oriana', web);
  }

  if (!(await probe(`${WEB}/api/health`, 'membership-rag-signing-proxy'))) {
    throw new Error('The Angular page is not connected to the signing proxy.');
  }

  console.log(`[dev] Oriana is ready at ${WEB}`);
} catch (error) {
  stop();
  console.error(`[dev] ${error instanceof Error ? error.message : error}`);
  process.exitCode = 1;
}
