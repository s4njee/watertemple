import { defineConfig } from 'vitest/config';
export default defineConfig({ test: { environment: 'node', include: ['experiments/**/*.test.ts'], setupFiles: ['tests/setup.ts'], maxWorkers: 1 } });
