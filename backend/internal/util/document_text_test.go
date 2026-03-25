package util

import (
	"strings"
	"testing"
)

func TestSemanticChunkBasic(t *testing.T) {
	text := "第一段第一句。第一段第二句。\n\n第二段第一句。"
	cfg := DefaultSemanticChunkConfig()
	cfg.MaxChunkSize = 20
	cfg.MinChunkSize = 1
	cfg.OverlapSize = 0

	chunks := SemanticChunk(text, cfg)
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(chunks))
	}
	if chunks[0] != "第一段第一句。第一段第二句。" {
		t.Fatalf("unexpected first chunk: %q", chunks[0])
	}
	if chunks[len(chunks)-1] != "第二段第一句。" {
		t.Fatalf("unexpected last chunk: %q", chunks[len(chunks)-1])
	}
}

func TestSemanticChunkOverlap(t *testing.T) {
	text := "句子一。句子二。句子三。"
	cfg := DefaultSemanticChunkConfig()
	cfg.MaxChunkSize = 6
	cfg.MinChunkSize = 1
	cfg.OverlapSize = 2

	chunks := SemanticChunk(text, cfg)
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(chunks))
	}
	prevTail := []rune(strings.TrimSpace(chunks[0]))
	if len(prevTail) < cfg.OverlapSize {
		t.Fatalf("expected chunk length >= overlap")
	}
	prefix := string(prevTail[len(prevTail)-cfg.OverlapSize:])
	if !strings.HasPrefix(chunks[1], prefix) {
		t.Fatalf("expected overlap prefix %q, got %q", prefix, chunks[1])
	}
}

func TestSemanticChunkLongSentence(t *testing.T) {
	text := strings.Repeat("很长", 40) + "。"
	cfg := DefaultSemanticChunkConfig()
	cfg.MaxChunkSize = 10
	cfg.MinChunkSize = 1
	cfg.OverlapSize = 0

	chunks := SemanticChunk(text, cfg)
	if len(chunks) < 2 {
		t.Fatalf("expected forced split, got %d", len(chunks))
	}
	for i, chunk := range chunks {
		if len([]rune(chunk)) > cfg.MaxChunkSize {
			t.Fatalf("chunk %d too long: %d", i, len([]rune(chunk)))
		}
	}
}

func TestSemanticChunkMinSize(t *testing.T) {
	text := "短句。\n\n这是一个足够长的句子，用于验证最小长度过滤。"
	cfg := DefaultSemanticChunkConfig()
	cfg.MaxChunkSize = 50
	cfg.MinChunkSize = 10
	cfg.OverlapSize = 0

	chunks := SemanticChunk(text, cfg)
	if len(chunks) != 1 {
		t.Fatalf("expected short chunk to be filtered, got %d", len(chunks))
	}
	if !strings.Contains(chunks[0], "这是一个足够长的句子") {
		t.Fatalf("unexpected chunk: %q", chunks[0])
	}
}
