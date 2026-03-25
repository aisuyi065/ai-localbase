import React from 'react'
import { AppConfig, ChatConfig, EmbeddingConfig } from '../../App'

interface SettingsPanelProps {
  config: AppConfig
  onClose: () => void
  onChatConfigChange: <K extends keyof ChatConfig>(key: K, value: ChatConfig[K]) => void
  onEmbeddingConfigChange: <K extends keyof EmbeddingConfig>(
    key: K,
    value: EmbeddingConfig[K],
  ) => void
}

const SettingsPanel: React.FC<SettingsPanelProps> = ({
  config,
  onClose,
  onChatConfigChange,
  onEmbeddingConfigChange,
}) => {
  return (
    <div className="settings-modal-backdrop" onClick={onClose}>
      <div className="settings-modal settings-modal-single" onClick={(event) => event.stopPropagation()}>
        <div className="settings-modal-header">
          <div>
            <h3>AI 设置</h3>
            <p>分别管理聊天模型与 Embedding 模型配置</p>
          </div>
          <button type="button" className="ghost-btn settings-close-btn" onClick={onClose}>
            关闭
          </button>
        </div>

        <div className="settings-modal-scroll">
          <section className="settings-panel-block ai-config-panel single-column">
            <div className="section-title-row knowledge-panel-header">
              <h3>聊天模型</h3>
            </div>

            <div className="ai-config-fields">
              <label className="settings-field">
                <span>Provider</span>
                <select
                  value={config.chat.provider}
                  onChange={(event) =>
                    onChatConfigChange('provider', event.target.value as ChatConfig['provider'])
                  }
                >
                  <option value="ollama">Ollama</option>
                  <option value="openai-compatible">OpenAI Compatible</option>
                </select>
              </label>

              <label className="settings-field">
                <span>Base URL</span>
                <input
                  value={config.chat.baseUrl}
                  onChange={(event) => onChatConfigChange('baseUrl', event.target.value)}
                  placeholder={
                    config.chat.provider === 'ollama'
                      ? 'http://localhost:11434'
                      : 'http://localhost:11434/v1'
                  }
                />
              </label>

              <label className="settings-field">
                <span>Model</span>
                <input
                  value={config.chat.model}
                  onChange={(event) => onChatConfigChange('model', event.target.value)}
                  placeholder="llama3.2"
                />
              </label>

              <label className="settings-field">
                <span>API Key</span>
                <input
                  type="password"
                  value={config.chat.apiKey}
                  onChange={(event) => onChatConfigChange('apiKey', event.target.value)}
                  placeholder="选填"
                />
              </label>

              <label className="settings-field settings-field-full">
                <span>Temperature: {config.chat.temperature.toFixed(1)}</span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={config.chat.temperature}
                  onChange={(event) =>
                    onChatConfigChange('temperature', Number(event.target.value))
                  }
                />
              </label>

              <label className="settings-field settings-field-full">
                <span>上下文消息数量</span>
                <input
                  type="number"
                  min="1"
                  max="100"
                  value={config.chat.contextMessageLimit}
                  onChange={(event) =>
                    onChatConfigChange('contextMessageLimit', Number(event.target.value))
                  }
                  placeholder="12"
                />
                <small>限制每次发送给模型的最近消息条数，范围 1-100。</small>
              </label>
            </div>
          </section>

          <section className="settings-panel-block ai-config-panel single-column">
            <div className="section-title-row knowledge-panel-header">
              <h3>Embedding 模型</h3>
            </div>

            <div className="ai-config-fields">
              <label className="settings-field">
                <span>Provider</span>
                <select
                  value={config.embedding.provider}
                  onChange={(event) =>
                    onEmbeddingConfigChange(
                      'provider',
                      event.target.value as EmbeddingConfig['provider'],
                    )
                  }
                >
                  <option value="ollama">Ollama</option>
                  <option value="openai-compatible">OpenAI Compatible</option>
                </select>
              </label>

              <label className="settings-field">
                <span>Base URL</span>
                <input
                  value={config.embedding.baseUrl}
                  onChange={(event) => onEmbeddingConfigChange('baseUrl', event.target.value)}
                  placeholder={
                    config.embedding.provider === 'ollama'
                      ? 'http://localhost:11434'
                      : 'http://localhost:11434/v1'
                  }
                />
              </label>

              <label className="settings-field">
                <span>Model</span>
                <input
                  value={config.embedding.model}
                  onChange={(event) => onEmbeddingConfigChange('model', event.target.value)}
                  placeholder="nomic-embed-text"
                />
              </label>

              <label className="settings-field">
                <span>API Key</span>
                <input
                  type="password"
                  value={config.embedding.apiKey}
                  onChange={(event) => onEmbeddingConfigChange('apiKey', event.target.value)}
                  placeholder="选填"
                />
              </label>
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}

export default SettingsPanel
