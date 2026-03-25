package service

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"ai-localbase/internal/model"
)

type LLMService struct {
	client *http.Client
}

// ── OpenAI-compatible structs ────────────────────────────────────────────────

type openAIChatRequest struct {
	Model       string              `json:"model"`
	Messages    []model.ChatMessage `json:"messages"`
	Temperature float64             `json:"temperature,omitempty"`
}

type openAIChatResponse struct {
	ID      string                        `json:"id"`
	Object  string                        `json:"object"`
	Created int64                         `json:"created"`
	Model   string                        `json:"model"`
	Choices []model.ChatCompletionChoice  `json:"choices"`
	Error   *openAICompatibleErrorPayload `json:"error,omitempty"`
}

type openAICompatibleErrorPayload struct {
	Message string `json:"message"`
	Type    string `json:"type,omitempty"`
	Code    any    `json:"code,omitempty"`
}

type openAIChatStreamRequest struct {
	Model       string              `json:"model"`
	Messages    []model.ChatMessage `json:"messages"`
	Temperature float64             `json:"temperature,omitempty"`
	Stream      bool                `json:"stream"`
}

type openAIChatStreamChunk struct {
	Choices []struct {
		Delta struct {
			Role    string `json:"role,omitempty"`
			Content string `json:"content,omitempty"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason,omitempty"`
	} `json:"choices"`
	Error *openAICompatibleErrorPayload `json:"error,omitempty"`
}

// ── Ollama native API structs ────────────────────────────────────────────────

type ollamaChatRequest struct {
	Model    string              `json:"model"`
	Messages []model.ChatMessage `json:"messages"`
	Stream   bool                `json:"stream"`
	Options  *ollamaOptions      `json:"options,omitempty"`
}

type ollamaOptions struct {
	Temperature float64 `json:"temperature,omitempty"`
}

type ollamaChatResponse struct {
	Model     string            `json:"model"`
	CreatedAt string            `json:"created_at"`
	Message   model.ChatMessage `json:"message"`
	Done      bool              `json:"done"`
	Error     string            `json:"error,omitempty"`
}

// ── Constructor ──────────────────────────────────────────────────────────────

func NewLLMService() *LLMService {
	return &LLMService{
		client: &http.Client{Timeout: 90 * time.Second},
	}
}

// ── Public methods ───────────────────────────────────────────────────────────

func (s *LLMService) Chat(req model.ChatCompletionRequest) (model.ChatCompletionResponse, error) {
	cfg, err := normalizeChatConfig(req)
	if err != nil {
		return model.ChatCompletionResponse{}, err
	}

	var result model.ChatCompletionResponse
	if cfg.Provider == "ollama" {
		result, err = s.ollamaChat(cfg, req)
	} else {
		result, err = s.openAIChat(cfg, req)
	}

	if err != nil {
		fallbackContent := buildModelFallbackMessage(req)
		return model.ChatCompletionResponse{
			ID:      "chatcmpl-fallback",
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   cfg.Model,
			Choices: []model.ChatCompletionChoice{{
				Index: 0,
				Message: model.ChatMessage{
					Role:    "assistant",
					Content: fallbackContent,
				},
			}},
			Metadata: map[string]any{
				"degraded":         true,
				"fallbackStrategy": "local-message",
				"upstreamError":    err.Error(),
			},
		}, nil
	}

	return result, nil
}

func (s *LLMService) StreamChat(req model.ChatCompletionRequest, onChunk func(string) error) error {
	cfg, err := normalizeChatConfig(req)
	if err != nil {
		return err
	}

	if cfg.Provider == "ollama" {
		err = s.ollamaStreamChat(cfg, req, onChunk)
	} else {
		err = s.openAIStreamChat(cfg, req, onChunk)
	}

	if err != nil {
		fallbackContent := buildModelFallbackMessage(req)
		return onChunk(fallbackContent)
	}

	return nil
}

// ── OpenAI-compatible implementation ─────────────────────────────────────────

func (s *LLMService) openAIChat(cfg model.ChatModelConfig, req model.ChatCompletionRequest) (model.ChatCompletionResponse, error) {
	payload := openAIChatRequest{
		Model:       cfg.Model,
		Messages:    req.Messages,
		Temperature: cfg.Temperature,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return model.ChatCompletionResponse{}, fmt.Errorf("failed to encode chat request")
	}

	endpoint := strings.TrimRight(cfg.BaseURL, "/") + "/chat/completions"
	var result model.ChatCompletionResponse
	err = retryWithBackoff(context.Background(), 3, 250*time.Millisecond, func() error {
		httpReq, err := http.NewRequest(http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			return fmt.Errorf("failed to create model request")
		}

		httpReq.Header.Set("Content-Type", "application/json")
		if cfg.APIKey != "" {
			httpReq.Header.Set("Authorization", "Bearer "+cfg.APIKey)
		}

		resp, err := s.client.Do(httpReq)
		if err != nil {
			return fmt.Errorf("failed to call model api: %w", err)
		}
		defer resp.Body.Close()

		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			return fmt.Errorf("failed to read model response")
		}

		var llmResp openAIChatResponse
		if err := json.Unmarshal(respBody, &llmResp); err != nil {
			return fmt.Errorf("invalid model response format")
		}

		if resp.StatusCode >= http.StatusBadRequest {
			if llmResp.Error != nil && strings.TrimSpace(llmResp.Error.Message) != "" {
				return fmt.Errorf("model api error: %s", llmResp.Error.Message)
			}
			return fmt.Errorf("model api error: http %d", resp.StatusCode)
		}

		if len(llmResp.Choices) == 0 {
			return fmt.Errorf("model api returned empty choices")
		}

		result = model.ChatCompletionResponse{
			ID:      llmResp.ID,
			Object:  llmResp.Object,
			Created: llmResp.Created,
			Model:   llmResp.Model,
			Choices: llmResp.Choices,
		}
		return nil
	})

	return result, err
}

func (s *LLMService) openAIStreamChat(cfg model.ChatModelConfig, req model.ChatCompletionRequest, onChunk func(string) error) error {
	payload := openAIChatStreamRequest{
		Model:       cfg.Model,
		Messages:    req.Messages,
		Temperature: cfg.Temperature,
		Stream:      true,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to encode chat request")
	}

	endpoint := strings.TrimRight(cfg.BaseURL, "/") + "/chat/completions"
	return retryWithBackoff(context.Background(), 2, 200*time.Millisecond, func() error {
		httpReq, err := http.NewRequest(http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			return fmt.Errorf("failed to create model request")
		}

		httpReq.Header.Set("Content-Type", "application/json")
		if cfg.APIKey != "" {
			httpReq.Header.Set("Authorization", "Bearer "+cfg.APIKey)
		}

		resp, err := s.client.Do(httpReq)
		if err != nil {
			return fmt.Errorf("failed to call model api: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode >= http.StatusBadRequest {
			respBody, readErr := io.ReadAll(resp.Body)
			if readErr != nil {
				return fmt.Errorf("model api error: http %d", resp.StatusCode)
			}

			var llmResp openAIChatResponse
			if err := json.Unmarshal(respBody, &llmResp); err == nil && llmResp.Error != nil && strings.TrimSpace(llmResp.Error.Message) != "" {
				return fmt.Errorf("model api error: %s", llmResp.Error.Message)
			}

			return fmt.Errorf("model api error: http %d", resp.StatusCode)
		}

		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" || !strings.HasPrefix(line, "data:") {
				continue
			}

			payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if payload == "[DONE]" {
				break
			}

			var chunk openAIChatStreamChunk
			if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
				continue
			}

			if chunk.Error != nil && strings.TrimSpace(chunk.Error.Message) != "" {
				return fmt.Errorf("model api error: %s", chunk.Error.Message)
			}

			for _, choice := range chunk.Choices {
				if strings.TrimSpace(choice.Delta.Content) == "" {
					continue
				}
				if err := onChunk(choice.Delta.Content); err != nil {
					return err
				}
			}
		}

		if err := scanner.Err(); err != nil {
			return fmt.Errorf("failed to read model stream")
		}

		return nil
	})
}

// ── Ollama native implementation ──────────────────────────────────────────────

func (s *LLMService) ollamaChat(cfg model.ChatModelConfig, req model.ChatCompletionRequest) (model.ChatCompletionResponse, error) {
	payload := ollamaChatRequest{
		Model:    cfg.Model,
		Messages: req.Messages,
		Stream:   false,
	}
	if cfg.Temperature > 0 {
		payload.Options = &ollamaOptions{Temperature: cfg.Temperature}
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return model.ChatCompletionResponse{}, fmt.Errorf("failed to encode chat request")
	}

	endpoint := strings.TrimRight(cfg.BaseURL, "/") + "/api/chat"
	var result model.ChatCompletionResponse
	err = retryWithBackoff(context.Background(), 3, 250*time.Millisecond, func() error {
		httpReq, err := http.NewRequest(http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			return fmt.Errorf("failed to create model request")
		}
		httpReq.Header.Set("Content-Type", "application/json")

		resp, err := s.client.Do(httpReq)
		if err != nil {
			return fmt.Errorf("failed to call model api: %w", err)
		}
		defer resp.Body.Close()

		respBody, err := io.ReadAll(resp.Body)
		if err != nil {
			return fmt.Errorf("failed to read model response")
		}

		var ollamaResp ollamaChatResponse
		if err := json.Unmarshal(respBody, &ollamaResp); err != nil {
			return fmt.Errorf("invalid model response format")
		}

		if resp.StatusCode >= http.StatusBadRequest {
			if strings.TrimSpace(ollamaResp.Error) != "" {
				return fmt.Errorf("model api error: %s", ollamaResp.Error)
			}
			return fmt.Errorf("model api error: http %d", resp.StatusCode)
		}

		if strings.TrimSpace(ollamaResp.Message.Content) == "" {
			return fmt.Errorf("model api returned empty response")
		}

		result = model.ChatCompletionResponse{
			ID:      "ollama-" + ollamaResp.Model,
			Object:  "chat.completion",
			Created: time.Now().Unix(),
			Model:   ollamaResp.Model,
			Choices: []model.ChatCompletionChoice{{
				Index:   0,
				Message: ollamaResp.Message,
			}},
		}
		return nil
	})

	return result, err
}

func (s *LLMService) ollamaStreamChat(cfg model.ChatModelConfig, req model.ChatCompletionRequest, onChunk func(string) error) error {
	payload := ollamaChatRequest{
		Model:    cfg.Model,
		Messages: req.Messages,
		Stream:   true,
	}
	if cfg.Temperature > 0 {
		payload.Options = &ollamaOptions{Temperature: cfg.Temperature}
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to encode chat request")
	}

	endpoint := strings.TrimRight(cfg.BaseURL, "/") + "/api/chat"
	return retryWithBackoff(context.Background(), 2, 200*time.Millisecond, func() error {
		httpReq, err := http.NewRequest(http.MethodPost, endpoint, bytes.NewReader(body))
		if err != nil {
			return fmt.Errorf("failed to create model request")
		}
		httpReq.Header.Set("Content-Type", "application/json")

		resp, err := s.client.Do(httpReq)
		if err != nil {
			return fmt.Errorf("failed to call model api: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode >= http.StatusBadRequest {
			respBody, readErr := io.ReadAll(resp.Body)
			if readErr != nil {
				return fmt.Errorf("model api error: http %d", resp.StatusCode)
			}
			var ollamaResp ollamaChatResponse
			if err := json.Unmarshal(respBody, &ollamaResp); err == nil && strings.TrimSpace(ollamaResp.Error) != "" {
				return fmt.Errorf("model api error: %s", ollamaResp.Error)
			}
			return fmt.Errorf("model api error: http %d", resp.StatusCode)
		}

		// Ollama streams newline-delimited JSON objects (NDJSON), not SSE
		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}

			var chunk ollamaChatResponse
			if err := json.Unmarshal([]byte(line), &chunk); err != nil {
				continue
			}

			if strings.TrimSpace(chunk.Error) != "" {
				return fmt.Errorf("model api error: %s", chunk.Error)
			}

			if chunk.Done {
				break
			}

			if content := strings.TrimSpace(chunk.Message.Content); content != "" {
				if err := onChunk(content); err != nil {
					return err
				}
			}
		}

		if err := scanner.Err(); err != nil {
			return fmt.Errorf("failed to read model stream")
		}

		return nil
	})
}

// ── Helpers ───────────────────────────────────────────────────────────────────

func normalizeChatConfig(req model.ChatCompletionRequest) (model.ChatModelConfig, error) {
	cfg := req.Config
	if strings.TrimSpace(cfg.Model) == "" {
		return model.ChatModelConfig{}, fmt.Errorf("model is required")
	}
	if strings.TrimSpace(cfg.BaseURL) == "" {
		if cfg.Provider == "ollama" {
			cfg.BaseURL = "http://localhost:11434"
		} else {
			cfg.BaseURL = "http://localhost:11434/v1"
		}
	}
	if cfg.Temperature <= 0 {
		cfg.Temperature = 0.7
	}
	if strings.TrimSpace(cfg.Provider) == "" {
		cfg.Provider = "ollama"
	}
	return cfg, nil
}

func buildModelFallbackMessage(req model.ChatCompletionRequest) string {
	modelName := strings.TrimSpace(req.Config.Model)
	if modelName == "" {
		modelName = strings.TrimSpace(req.Model)
	}
	hint := ""
	if modelName != "" {
		hint = fmt.Sprintf("请检查模型 **%s** 是否已在 Ollama 中下载（`ollama pull %s`），或在设置中更换为可用模型。", modelName, modelName)
	} else {
		hint = "请在设置中配置正确的 Chat 模型。"
	}
	return fmt.Sprintf("⚠️ AI 模型调用失败\n\n%s", hint)
}
