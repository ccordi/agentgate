// k6 load script for the agentgate benchmark (ARTIFACT 3).
//
// Drives a streaming OpenAI chat-completion against either the gateway or the mock upstream
// directly (baseline), parameterized entirely by env so bench/run.py can reuse one script:
//
//   TARGET_URL   full POST URL (gateway :4300 or mock :4200)
//   MODE         "rate" (constant-arrival-rate, default) | "vus" (constant-vus)
//   RATE         req/s for rate mode (default 100)
//   VUS          virtual users for vus mode (default 20)
//   DURATION     test window (default 15s)
//   PRE_VUS      preallocated VUs for rate mode (default 120)
//   PAYLOAD      "small" (default) | "large" (adds a 4 KB tool-output block → inject cost)
//   SUMMARY_OUT  path to write the summary JSON (handleSummary)
//
// k6 buffers the full response body, so http_req_waiting ≈ time-to-first-byte (SSE TTFB)
// and http_req_duration ≈ full-stream time → streaming overhead = duration − waiting.

import http from 'k6/http';
import { check } from 'k6';
import { Trend } from 'k6/metrics';

const TARGET = __ENV.TARGET_URL || 'http://127.0.0.1:4300/v1/chat/completions';
const MODE = __ENV.MODE || 'rate';
const RATE = parseInt(__ENV.RATE || '100', 10);
const VUS = parseInt(__ENV.VUS || '20', 10);
const DURATION = __ENV.DURATION || '15s';
const PRE_VUS = parseInt(__ENV.PRE_VUS || '120', 10);
const PAYLOAD = __ENV.PAYLOAD || 'small';

const ttfb = new Trend('sse_ttfb_ms', true);
const total = new Trend('sse_total_ms', true);

export const options = {
  scenarios:
    MODE === 'vus'
      ? { load: { executor: 'constant-vus', vus: VUS, duration: DURATION } }
      : {
          load: {
            executor: 'constant-arrival-rate',
            rate: RATE,
            timeUnit: '1s',
            duration: DURATION,
            preAllocatedVUs: PRE_VUS,
            maxVUs: PRE_VUS * 4,
          },
        },
  thresholds: { http_req_failed: ['rate<0.01'] },
  // Ensure p99 (and the rest) are present in the handleSummary export.
  summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
};

const BIG_TOOL = 'A'.repeat(4000);

function payload() {
  const messages = [{ role: 'user', content: 'summarize this article' }];
  if (PAYLOAD === 'large') {
    messages.push({ role: 'tool', tool_call_id: 'c1', content: 'Page text. ' + BIG_TOOL });
  }
  return JSON.stringify({
    model: 'm',
    stream: true,
    stream_options: { include_usage: true },
    messages,
  });
}

const PARAMS = {
  headers: { 'Content-Type': 'application/json', Authorization: 'Bearer benchtoken' },
};

export default function () {
  const res = http.post(TARGET, payload(), PARAMS);
  ttfb.add(res.timings.waiting);
  total.add(res.timings.duration);
  check(res, {
    'status 200': (r) => r.status === 200,
    'streamed [DONE]': (r) => r.body && r.body.indexOf('[DONE]') !== -1,
  });
}

export function handleSummary(data) {
  const out = __ENV.SUMMARY_OUT || 'bench_summary.json';
  return { [out]: JSON.stringify(data) };
}
