# Product Roadmap

> Last Updated: 2025-09-16
> Version: 1.0.0
> Status: Planning

## Phase 0: Already Completed

The following features have been implemented:

- [x] **Basic inference script** - Command-line interface for model interaction `XS`
- [x] **Chain-of-thought reasoning support** - Parse and display thinking tokens separately `S`
- [x] **Interactive chat mode** - Real-time conversation with the model `XS`
- [x] **Memory-efficient loading** - GPU memory management and CPU offloading `S`
- [x] **Dependency management** - Requirements and testing scripts `XS`

## Phase 1: OpenAI-Compatible API Server (2-3 weeks)

**Goal:** Provide drop-in replacement for OpenAI API
**Success Criteria:** Successfully serve requests compatible with OpenAI client libraries

### Must-Have Features

- [ ] **FastAPI server implementation** - RESTful API with OpenAI-compatible endpoints `M`
- [ ] **Streaming response support** - Server-sent events for real-time generation `S`
- [ ] **Request queuing system** - Handle multiple concurrent requests `M`
- [ ] **API key authentication** - Basic security for production use `S`
- [ ] **Docker containerization** - Easy deployment package `S`

### Dependencies

- FastAPI framework
- Async request handling

## Phase 2: Performance & Testing (1-2 weeks)

**Goal:** Ensure reliability and optimize performance
**Success Criteria:** Comprehensive test coverage and documented benchmarks

### Must-Have Features

- [ ] **Unit test suite** - pytest-based testing for all components `M`
- [ ] **Integration tests** - End-to-end API testing `S`
- [ ] **Performance benchmarking** - Token/sec metrics across hardware `S`
- [ ] **Memory profiling** - Optimize GPU/CPU memory usage `M`
- [ ] **Load testing** - Determine concurrent request limits `S`

### Dependencies

- pytest framework
- Benchmark suite implementation

## Phase 3: Enhanced Features (2-3 weeks)

**Goal:** Add advanced capabilities for power users
**Success Criteria:** Feature parity with production inference servers

### Must-Have Features

- [ ] **Batch inference support** - Process multiple prompts efficiently `L`
- [ ] **Model parameter tuning** - Runtime adjustment of generation parameters `M`
- [ ] **Conversation management** - Multi-turn chat with context handling `M`
- [ ] **Response caching** - Reduce redundant computations `S`
- [ ] **Metrics dashboard** - Web UI for monitoring `M`

### Dependencies

- Phase 1 completion
- Extended API design

## Phase 4: Deprecation Planning (1 week)

**Goal:** Graceful transition when llama.cpp supports Qwen3-Next
**Success Criteria:** Users smoothly migrate to llama.cpp

### Must-Have Features

- [ ] **Migration guide** - Documentation for transitioning to llama.cpp `S`
- [ ] **Feature comparison** - Document differences between implementations `S`
- [ ] **Data export tools** - Export conversation history and settings `S`
- [ ] **Deprecation notices** - Clear communication of sunset timeline `XS`

### Dependencies

- llama.cpp Qwen3-Next support
- Community feedback