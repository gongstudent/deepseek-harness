import { createServer } from 'node:http'
import type { IncomingMessage, Server, ServerResponse } from 'node:http'
import { afterEach, describe, expect, it } from 'vitest'
import { resolveProfiles } from '../src/config.ts'
import { LocalRouteServer } from '../src/local-route.ts'

const servers: Server[] = []
const routes: LocalRouteServer[] = []

afterEach(async () => {
  await Promise.all(routes.splice(0).map(route => route.close()))
  await Promise.all(servers.splice(0).map(server => new Promise<void>(resolve => server.close(() => { resolve() }))))
})

type TestHandler = (request: IncomingMessage, response: ServerResponse) => void | Promise<void>

async function upstream(handler: TestHandler): Promise<{ server: Server; baseURL: string }> {
  const server = createServer((request, response) => {
    void Promise.resolve(handler(request, response)).catch((error: unknown) => {
      response.destroy(error instanceof Error ? error : new Error(String(error)))
    })
  })
  servers.push(server)
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => { resolve() })
  })
  const address = server.address()
  if (typeof address !== 'object' || address === null) throw new Error('test upstream did not bind TCP')
  return { server, baseURL: `http://127.0.0.1:${String(address.port)}` }
}

describe('local route HTTP proxy', () => {
  it('converts Responses requests to Anthropic and applies header/body overrides', async () => {
    let received: {
      url?: string | undefined
      headers?: Record<string, string | string[] | undefined>
      body?: unknown
    } = {}
    const target = await upstream(async (request, response) => {
      const chunks: Buffer[] = []
      for await (const chunk of request) chunks.push(Buffer.from(chunk as Uint8Array))
      received = { url: request.url, headers: request.headers, body: JSON.parse(Buffer.concat(chunks).toString('utf8')) as unknown }
      response.writeHead(200, { 'content-type': 'application/json' })
      response.end(JSON.stringify({
        id: 'msg_1',
        type: 'message',
        role: 'assistant',
        model: 'route-model',
        content: [{ type: 'text', text: 'converted answer' }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 4, output_tokens: 2 },
      }))
    })
    const profiles = resolveProfiles({
      gateway: {
        apiKeyEnv: 'GATEWAY_API_KEY',
        api: 'anthropic-messages',
        inboundApi: 'openai-responses',
        baseURL: target.baseURL,
        headers: { 'X-Route': 'override' },
        bodyOverrides: { temperature: 0.25, metadata: { source: 'local' } },
        models: [{ id: 'route-model', maxTokens: 2048 }],
      },
    })
    const route = new LocalRouteServer({ profiles: () => profiles, resolveApiKey: () => Promise.resolve('secret-key') })
    routes.push(route)
    await route.configure({ enabled: true, port: 0 })
    const response = await fetch(`http://127.0.0.1:${String(route.port)}/v1/responses`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-dsh-provider': 'gateway' },
      body: JSON.stringify({ model: 'route-model', input: 'hello' }),
    })
    expect(response.status).toBe(200)
    const body = await response.json() as { object: string; output_text: string; usage: Record<string, number> }
    expect(body).toMatchObject({ object: 'response', output_text: 'converted answer' })
    expect(body.usage).toEqual({ input_tokens: 4, output_tokens: 2, total_tokens: 6 })
    expect(received.url).toBe('/v1/messages')
    expect(received.headers?.['x-api-key']).toBe('secret-key')
    expect(received.headers?.['x-route']).toBe('override')
    expect(received.body).toMatchObject({
      model: 'route-model',
      messages: [{ role: 'user', content: [{ type: 'text', text: 'hello' }] }],
      max_tokens: 2048,
      temperature: 0.25,
      metadata: { source: 'local' },
      stream: false,
    })

    const streamed = await fetch(`http://127.0.0.1:${String(route.port)}/v1/responses`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-dsh-provider': 'gateway' },
      body: JSON.stringify({ model: 'route-model', input: 'hello', stream: true }),
    })
    expect(streamed.headers.get('content-type')).toContain('text/event-stream')
    const events = await streamed.text()
    expect(events).toContain('event: response.output_text.delta')
    expect(events).toContain('converted answer')
    expect(received.body).toMatchObject({ stream: false })
  })

  it('passes same-protocol SSE through and stops listening when disabled', async () => {
    const target = await upstream((_request, response) => {
      response.writeHead(200, { 'content-type': 'text/event-stream' })
      response.end('data: {"id":"chunk-1"}\n\ndata: [DONE]\n\n')
    })
    const profiles = resolveProfiles({
      openai: {
        api: 'openai-completions',
        inboundApi: 'openai-completions',
        baseURL: target.baseURL,
        models: [{ id: 'same-model' }],
      },
    })
    const route = new LocalRouteServer({ profiles: () => profiles, resolveApiKey: () => Promise.resolve(undefined) })
    routes.push(route)
    await route.configure({ enabled: true, port: 0 })
    const port = route.port
    if (port === undefined) throw new Error('local route did not bind TCP')
    const response = await fetch(`http://127.0.0.1:${String(port)}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: 'same-model', messages: [{ role: 'user', content: 'hello' }], stream: true }),
    })
    expect(await response.text()).toBe('data: {"id":"chunk-1"}\n\ndata: [DONE]\n\n')

    const occupiedPort = Number(new URL(target.baseURL).port)
    await expect(route.configure({ enabled: true, port: occupiedPort })).rejects.toThrow(/EADDRINUSE/)
    expect(route.port).toBe(port)
    expect((await fetch(`http://127.0.0.1:${String(port)}/health`)).status).toBe(200)

    await route.configure({ enabled: false, port })
    expect(route.port).toBeUndefined()
    await expect(fetch(`http://127.0.0.1:${String(port)}/health`)).rejects.toThrow()
  })

  it('answers an unserviceable request with its own status, not a gateway failure', async () => {
    const target = await upstream((_request, response) => {
      response.writeHead(500)
      response.end()
    })
    const profile = {
      api: 'openai-completions',
      inboundApi: 'openai-completions',
      baseURL: target.baseURL,
      models: [{ id: 'shared-model' }],
    }
    const profiles = resolveProfiles({ first: profile, second: profile })
    const route = new LocalRouteServer({ profiles: () => profiles, resolveApiKey: () => Promise.resolve(undefined) })
    routes.push(route)
    await route.configure({ enabled: true, port: 0 })
    const post = (body: unknown): Promise<Response> => fetch(`http://127.0.0.1:${String(route.port)}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    })

    const ambiguous = await post({ model: 'shared-model', messages: [] })
    expect(ambiguous.status).toBe(400)
    expect(await ambiguous.text()).toContain('x-dsh-provider')

    const unknown = await post({ model: 'absent-model', messages: [] })
    expect(unknown.status).toBe(404)
    expect(await unknown.text()).toContain('No local route exposes model')

    const modelless = await post({ messages: [] })
    expect(modelless.status).toBe(400)
  })

  it('numbers synthetic tool-call deltas so a streaming client can accumulate them', async () => {
    const target = await upstream((_request, response) => {
      response.writeHead(200, { 'content-type': 'application/json' })
      response.end(JSON.stringify({
        id: 'msg_1',
        type: 'message',
        role: 'assistant',
        model: 'route-model',
        content: [
          { type: 'tool_use', id: 'toolu_a', name: 'bash', input: { command: 'ls' } },
          { type: 'tool_use', id: 'toolu_b', name: 'read', input: { path: 'a.txt' } },
        ],
        stop_reason: 'tool_use',
        usage: { input_tokens: 3, output_tokens: 1 },
      }))
    })
    const profiles = resolveProfiles({
      gateway: {
        api: 'anthropic-messages',
        inboundApi: 'openai-completions',
        baseURL: target.baseURL,
        models: [{ id: 'route-model', maxTokens: 1024 }],
      },
    })
    const route = new LocalRouteServer({ profiles: () => profiles, resolveApiKey: () => Promise.resolve(undefined) })
    routes.push(route)
    await route.configure({ enabled: true, port: 0 })
    const streamed = await fetch(`http://127.0.0.1:${String(route.port)}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: 'route-model', messages: [{ role: 'user', content: 'go' }], stream: true }),
    })
    const first = (await streamed.text()).split('\n\n')[0] ?? ''
    const chunk = JSON.parse(first.replace(/^data: /, '')) as {
      choices: Array<{ delta: { tool_calls: Array<{ index: number; id: string }> } }>
    }
    expect(chunk.choices[0]?.delta.tool_calls.map(call => [call.index, call.id])).toEqual([[0, 'toolu_a'], [1, 'toolu_b']])
  })
})
