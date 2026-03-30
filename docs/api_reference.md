# API Reference

MicroPDProxy exposes an OpenAI-compatible API surface. All endpoints are served
by the proxy process (default port `8868`).

## Authentication

Most endpoints are open. Admin endpoints (`/instances/add`, `/instances/remove`)
require an API key passed via the `X-API-Key` header. Set the key with the
`ADMIN_API_KEY` environment variable.

---

## Endpoints

### POST `/v1/chat/completions`

Chat completion (streaming and non-streaming). OpenAI-compatible.

**Request:**
```json
{
  "model": "DeepSeek-R1",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 256,
  "stream": false
}
```

**Response (non-streaming):**
```json
{
  "id": "cmpl-abc123",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Hello! How can I help?"},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18}
}
```

**Response (streaming, `stream: true`):**

Returns `text/event-stream` with SSE chunks:
```
data: {"id":"cmpl-abc123","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"index":0}]}

data: [DONE]
```

**Auth:** None required.

---

### POST `/v1/completions`

Text completion. OpenAI-compatible.

**Request:**
```json
{
  "model": "DeepSeek-R1",
  "prompt": "The meaning of life is",
  "max_tokens": 64,
  "stream": false
}
```

**Response:**
```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "choices": [
    {
      "index": 0,
      "text": " to find purpose and connection.",
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
}
```

**Auth:** None required.

---

### GET `/v1/models`

List available models. Aggregated from backend instances.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "DeepSeek-R1",
      "object": "model",
      "max_model_len": 65536
    }
  ]
}
```

**Auth:** None required.

---

### GET `/status`

Proxy status: lists prefill and decode node addresses and counts.

**Response:**
```json
{
  "prefill_instances": ["10.0.0.1:8100", "10.0.0.2:8100"],
  "decode_instances": ["10.0.0.3:8200", "10.0.0.4:8200"],
  "prefill_count": 2,
  "decode_count": 2
}
```

**Auth:** None required.

---

### GET `/health`

Health check across all backend nodes. Queries each node's `/health` endpoint.

**Response:**
```json
{
  "status": "healthy",
  "prefill": {"10.0.0.1:8100": "ok", "10.0.0.2:8100": "ok"},
  "decode": {"10.0.0.3:8200": "ok", "10.0.0.4:8200": "ok"}
}
```

**Auth:** None required.

---

### GET `/ping`

Simple liveness check for the proxy itself.

**Response:**
```
pong
```

**Auth:** None required.

---

### POST `/instances/add`

Dynamically add a prefill or decode instance.

**Request:**
```json
{
  "instance_type": "prefill",
  "instance": "10.0.0.5:8100"
}
```

**Response:**
```json
{
  "status": "added",
  "instance_type": "prefill",
  "instance": "10.0.0.5:8100"
}
```

**Auth:** Required. Pass `X-API-Key` header matching `ADMIN_API_KEY`.

---

### POST `/instances/remove`

Dynamically remove a prefill or decode instance.

**Request:**
```json
{
  "instance_type": "decode",
  "instance": "10.0.0.4:8200"
}
```

**Response:**
```json
{
  "status": "removed",
  "instance_type": "decode",
  "instance": "10.0.0.4:8200"
}
```

**Auth:** Required. Pass `X-API-Key` header matching `ADMIN_API_KEY`.

---

### POST `/tokenize`

Tokenize text using the proxy's loaded tokenizer.

**Request:**
```json
{
  "prompt": "Hello world"
}
```

**Response:** Proxied to a backend instance.

**Auth:** None required.

---

### POST `/detokenize`

Convert token IDs back to text.

**Request:**
```json
{
  "tokens": [9906, 1917]
}
```

**Response:** Proxied to a backend instance.

**Auth:** None required.

---

### GET `/version`

Return proxy version information.

**Response:**
```json
{
  "version": "0.1.0"
}
```

**Auth:** None required.
