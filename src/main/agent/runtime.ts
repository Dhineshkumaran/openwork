/* eslint-disable @typescript-eslint/no-unused-vars */
import { createDeepAgent, StateBackend, FilesystemBackend } from 'deepagents'
import type { BackendProtocol } from 'deepagents'
import type { BaseStore } from '@langchain/langgraph-checkpoint'
import { app } from 'electron'
import { join } from 'path'
import { getDefaultModel, getApiKey } from '../ipc/models'
import { ChatAnthropic } from '@langchain/anthropic'
import { ChatOpenAI } from '@langchain/openai'
import { SqlJsSaver } from '../checkpointer/sqljs-saver'

import type * as _lcTypes from 'langchain'
import type * as _lcMessages from '@langchain/core/messages'
import type * as _lcLanggraph from '@langchain/langgraph'
import type * as _lcZodTypes from '@langchain/core/utils/types'

// Local type definition (matches deepagents StateAndStore)
interface StateAndStore {
  state: unknown
  store?: BaseStore
  assistantId?: string
}

// Singleton checkpointer instance
let checkpointer: SqlJsSaver | null = null

export async function getCheckpointer(): Promise<SqlJsSaver> {
  if (!checkpointer) {
    const dbPath = join(app.getPath('userData'), 'langgraph.sqlite')
    checkpointer = new SqlJsSaver(dbPath)
    await checkpointer.initialize()
  }
  return checkpointer
}

// Get the appropriate model instance based on configuration
function getModelInstance(modelId?: string): ChatAnthropic | ChatOpenAI | string {
  const model = modelId || getDefaultModel()
  console.log('[Runtime] Using model:', model)

  // Determine provider from model ID
  if (model.startsWith('claude')) {
    const apiKey = getApiKey('anthropic')
    console.log('[Runtime] Anthropic API key present:', !!apiKey)
    if (!apiKey) {
      throw new Error('Anthropic API key not configured')
    }
    return new ChatAnthropic({
      model,
      anthropicApiKey: apiKey
    })
  } else if (model.startsWith('gpt')) {
    const apiKey = getApiKey('openai')
    console.log('[Runtime] OpenAI API key present:', !!apiKey)
    if (!apiKey) {
      throw new Error('OpenAI API key not configured')
    }
    return new ChatOpenAI({
      model,
      openAIApiKey: apiKey
    })
  } else if (model.startsWith('gemini')) {
    // For Gemini, we'd need @langchain/google-genai
    throw new Error('Gemini support coming soon')
  }

  // Default to model string (let deepagents handle it)
  return model
}

export interface CreateAgentRuntimeOptions {
  /** Model ID to use (defaults to configured default model) */
  modelId?: string
  /** Workspace path - if set, uses FilesystemBackend; otherwise uses StateBackend */
  workspacePath?: string | null
}

// Create agent runtime with configured model and checkpointer
export type AgentRuntime = ReturnType<typeof createDeepAgent>

/**
 * Create a backend factory that chooses between StateBackend and FilesystemBackend.
 *
 * - No workspacePath → StateBackend (virtual files in LangGraph state, isolated per thread)
 * - With workspacePath → FilesystemBackend (files on disk)
 */
function createBackendFactory(workspacePath: string | null) {
  return (stateAndStore: StateAndStore): BackendProtocol => {
    if (workspacePath) {
      console.log('[Runtime] Using FilesystemBackend at:', workspacePath)
      return new FilesystemBackend({
        rootDir: workspacePath,
        virtualMode: true // Use virtual paths starting with /
      })
    } else {
      console.log('[Runtime] Using StateBackend (virtual filesystem)')
      return new StateBackend(stateAndStore)
    }
  }
}

// eslint-disable-next-line @typescript-eslint/explicit-function-return-type
export async function createAgentRuntime(options: CreateAgentRuntimeOptions = {}) {
  const { modelId, workspacePath = null } = options

  console.log('[Runtime] Creating agent runtime...')

  const model = getModelInstance(modelId)
  console.log('[Runtime] Model instance created:', typeof model)

  const checkpointer = await getCheckpointer()
  console.log('[Runtime] Checkpointer ready')

  const agent = createDeepAgent({
    model,
    checkpointer,
    backend: createBackendFactory(workspacePath)
  })

  console.log(
    '[Runtime] Deep agent created with',
    workspacePath ? `FilesystemBackend at ${workspacePath}` : 'StateBackend (virtual)'
  )
  return agent
}

export type DeepAgent = ReturnType<typeof createDeepAgent>

// Clean up resources
export async function closeRuntime(): Promise<void> {
  if (checkpointer) {
    await checkpointer.close()
    checkpointer = null
  }
}
