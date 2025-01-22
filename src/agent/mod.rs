use anyhow::{anyhow, Result};
use serde_json::Value;
use std::collections::HashMap;
use tokio::time::timeout;

use crate::{
    llm::LLMClient,
    memory::{LongTermMemory, MemoryEntry, MemoryMetadata, ShortTermMemory},
    tools::Tool,
    types::{AgentConfig, AgentState, Decision, Message, Role, ToolExecutionResult},
};

pub struct Agent<M, H, L>
where
    M: LongTermMemory,
    H: ShortTermMemory,
    L: LLMClient,
{
    long_term_memory: M,
    short_term_memory: H,
    llm: L,
    tools: HashMap<String, Box<dyn Tool>>,
    config: AgentConfig,
    state: AgentState,
}

impl<M, H, L> Agent<M, H, L>
where
    M: LongTermMemory,
    H: ShortTermMemory,
    L: LLMClient,
{
    pub fn new(long_term_memory: M, short_term_memory: H, llm: L) -> Self {
        Self {
            long_term_memory,
            short_term_memory,
            llm,
            tools: HashMap::new(),
            config: AgentConfig::default(),
            state: AgentState::Ready,
        }
    }

    pub fn with_config(mut self, config: AgentConfig) -> Self {
        self.short_term_memory.add_message(
            Role::System,
            config.system_prompt.clone(),
        );
        self.config = config;
        self
    }

    pub fn register_tool<T: Tool + 'static>(&mut self, tool: T) {
        self.tools.insert(tool.name(), Box::new(tool));
    }

    pub async fn handle_message(&mut self, message: String) -> Result<String> {
        // 1. 状态检查
        if !matches!(self.state, AgentState::Ready) {
            return Err(anyhow!("Agent is not in ready state"));
        }
        self.state = AgentState::Processing;

        // 2. 添加用户消息到短期记忆
        self.short_term_memory.add_message(Role::User, message);

        // 3. 获取裁剪后的上下文
        let mut context = self.short_term_memory.get_context_messages(self.config.max_tokens);

        // 4. 循环处理直到得到最终响应
        let mut retries = 0;
        while retries < self.config.retry_config.max_retries {
            // 设置超时
            match timeout(self.config.timeout, self.get_decision(&context)).await {
                Ok(decision_result) => {
                    let decision = decision_result?;
                    match decision {
                        Decision::ExecuteTool(tool_args) => {
                            // 先记录 assistant 的决定
                            self.short_term_memory.add_message(
                                Role::Assistant,
                                format!("我需要使用 {} 工具来执行这个计算", tool_args.tool_name),
                            );

                            match self
                                .execute_tool(&tool_args.tool_name, tool_args.args)
                                .await?
                            {
                                ToolExecutionResult::Success { output, metadata } => {
                                    // 将工具执行结果作为工具消息添加到上下文
                                    self.short_term_memory.add_message(
                                        Role::Tool,
                                        format!("工具执行成功。结果是：{}", output),
                                    );
                                    // 获取新的上下文
                                    context = self.short_term_memory.get_context_messages(self.config.max_tokens);
                                    // 存储相关信息到长期记忆
                                    if let Some(metadata) = metadata {
                                        self.store_tool_result(&tool_args.tool_name, metadata)
                                            .await?;
                                    }
                                }
                                ToolExecutionResult::Failure {
                                    error,
                                    should_retry,
                                } => {
                                    if should_retry
                                        && retries < self.config.retry_config.max_retries
                                    {
                                        retries += 1;
                                        tokio::time::sleep(self.config.retry_config.retry_delay)
                                            .await;
                                        continue;
                                    }

                                    // 添加失败信息到上下文中，同时提供后续处理建议
                                    self.short_term_memory.add_message(
                                        Role::Tool,
                                        format!(
                                            "工具 {} 执行失败（错误信息：{}）。由于无法重试，请考虑使用其他方式解决问题或给出合适的响应。",
                                            tool_args.tool_name,
                                            error
                                        ),
                                    );
                                    context = self.short_term_memory.get_context_messages(self.config.max_tokens);
                                    continue;
                                }
                                ToolExecutionResult::NeedMoreInfo { missing_fields } => {
                                    self.state = AgentState::WaitingForUserInput;
                                    return Ok(format!(
                                        "需要更多信息: {}",
                                        missing_fields.join(", ")
                                    ));
                                }
                            }
                        }
                        Decision::Respond(response) => {
                            self.short_term_memory
                                .add_message(Role::Assistant, response.content.clone());
                            self.state = AgentState::Ready;
                            return Ok(response.content);
                        }
                        Decision::AskForClarification(request) => {
                            self.short_term_memory
                                .add_message(Role::Assistant, request.question.clone());
                            self.state = AgentState::WaitingForUserInput;
                            return Ok(request.question);
                        }
                        Decision::EndConversation { reason } => {
                            let response = reason.unwrap_or_else(|| "对话已结束".to_string());
                            self.short_term_memory
                                .add_message(Role::Assistant, response.clone());
                            self.state = AgentState::Terminated;
                            return Ok(response);
                        }
                    }
                }
                Err(err) => {
                    println!("running error: {}", err);
                    if retries < self.config.retry_config.max_retries {
                        retries += 1;
                        continue;
                    }
                    return Err(anyhow!("LLM request timed out"));
                }
            }
        }

        Err(anyhow!("超过最大重试次数"))
    }

    async fn get_decision(&self, messages: &[Message]) -> Result<Decision> {
        let tools: Vec<Box<dyn Tool>> = self.tools.values().map(|tool| tool.clone()).collect();

        self.llm
            .complete(messages, tools, self.config.max_tokens)
            .await
    }

    async fn execute_tool(&self, tool_name: &str, args: Value) -> Result<ToolExecutionResult> {
        let tool = self
            .tools
            .get(tool_name)
            .ok_or_else(|| anyhow!("Tool not found: {}", tool_name))?;

        tool.execute(args).await
    }

    async fn store_tool_result(&mut self, tool_name: &str, metadata: Value) -> Result<()> {
        let memory_entry = MemoryEntry {
            content: metadata.clone(),
            metadata: MemoryMetadata {
                timestamp: chrono::Utc::now(),
                tags: vec!["tool_result".to_string(), tool_name.to_string()],
                source: tool_name.to_string(),
            },
        };

        self.long_term_memory.store(memory_entry).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        llm::tests::MockLLMClient,
        memory::tests::{BasicShortTermMemory, MockLongTermMemory},
        tools::tests::EchoTool,
    };
    use pretty_assertions::assert_eq;
    use std::time::Duration;

    // 辅助函数: 创建一个测试用的Agent
    fn create_test_agent() -> Agent<MockLongTermMemory, BasicShortTermMemory, MockLLMClient> {
        let mut agent = Agent::new(
            MockLongTermMemory::new(),
            BasicShortTermMemory::new(),
            MockLLMClient::new(),
        );

        // 配置Agent
        let config = AgentConfig {
            system_prompt: "You are a helpful assistant.".to_string(),
            max_turns: 5,
            max_tokens: 1000,
            enable_parallel: false,
            retry_config: RetryConfig {
                max_retries: 2,
                retry_delay: Duration::from_millis(100),
                should_retry_on_error: true,
            },
            temperature: 0.7,
            timeout: Duration::from_secs(5),
        };
        agent = agent.with_config(config);

        // 注册工具
        agent.register_tool(EchoTool::new());
        agent
    }

    #[tokio::test]
    async fn test_agent_basic_flow() {
        let mut agent = create_test_agent();

        // 测试基本消息处理
        let response = agent.handle_message("Hello".to_string()).await.unwrap();
        assert_eq!(response, "Echo: Hello");
        assert!(matches!(agent.state, AgentState::Ready));

        // 验证短期记忆
        let context = agent.short_term_memory.get_context();
        assert_eq!(context.len(), 2); // user message + assistant response
        assert_eq!(context[0].role, Role::User);
        assert_eq!(context[0].content, "Hello");
        assert_eq!(context[1].role, Role::Assistant);
        assert_eq!(context[1].content, "Echo: Hello");
    }

    #[tokio::test]
    async fn test_agent_tool_execution() {
        let agent = create_test_agent();

        // 测试工具执行
        let result = agent
            .execute_tool("echo", serde_json::json!({"text": "test message"}))
            .await
            .unwrap();

        match result {
            ToolExecutionResult::Success { output, metadata } => {
                assert_eq!(output, "test message");
                assert!(metadata.is_none());
            }
            _ => panic!("Expected Success variant"),
        }

        // 测试工具不存在的情况
        let result = agent
            .execute_tool("nonexistent_tool", serde_json::json!({}))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_agent_memory_interaction() {
        let mut agent = create_test_agent();

        // 1. 添加一些消息到短期记忆
        agent
            .handle_message("First message".to_string())
            .await
            .unwrap();
        agent
            .handle_message("Second message".to_string())
            .await
            .unwrap();

        // 2. 验证短期记忆内容
        let context = agent.short_term_memory.get_context();
        assert_eq!(context.len(), 4); // 2*(user + assistant)

        // 3. 验证最近的对话
        let recent_messages = agent.short_term_memory.get_context();
        assert!(!recent_messages.is_empty());
        assert_eq!(recent_messages.last().unwrap().content, "Echo: Second message");

        // 4. 测试长期记忆存储和检索
        let test_data = serde_json::json!({
            "important_info": "test_data"
        });

        let memory_entry = MemoryEntry {
            content: test_data.clone(),
            metadata: MemoryMetadata {
                timestamp: chrono::Utc::now(),
                tags: vec!["test".to_string()],
                source: "test_tool".to_string(),
            },
        };

        agent.long_term_memory.store(memory_entry).await.unwrap();

        // 5. 通过语义查询验证记忆
        let results = agent
            .long_term_memory
            .recall(&MemoryQuery::Semantic {
                description: "test data".to_string(),
                limit: 1,
            })
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, test_data);
    }

    #[tokio::test]
    async fn test_agent_error_handling() {
        let mut agent = create_test_agent();

        // 1. 测试无效的工具调用
        let result = agent
            .execute_tool("invalid_tool", serde_json::json!({}))
            .await;
        assert!(result.is_err());

        // 2. 测试参数缺失的工具调用
        let result = agent.execute_tool("echo", serde_json::json!({})).await;
        assert!(result.is_err());

        // 3. 测试状态检查
        agent.state = AgentState::Processing;
        let result = agent.handle_message("Test".to_string()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_agent_state_transitions() {
        let mut agent = create_test_agent();

        // 1. 初始状态
        assert!(matches!(agent.state, AgentState::Ready));

        // 2. 处理消息时的状态转换
        let handle_future = agent.handle_message("Test".to_string());
        tokio::task::yield_now().await; // 让handle_message有机会开始执行

        // 3. 完成处理后的状态
        let _ = handle_future.await.unwrap();
        assert!(matches!(agent.state, AgentState::Ready));

        // 4. 错误状态
        agent.state = AgentState::Error("test error".to_string());
        let result = agent.handle_message("Test".to_string()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_agent_complex_conversation() {
        let mut agent = create_test_agent();

        // 1. 开始对话
        let response = agent.handle_message("Hello".to_string()).await.unwrap();
        assert_eq!(response, "Echo: Hello");

        // 2. 继续对话
        let response = agent
            .handle_message("How are you?".to_string())
            .await
            .unwrap();
        assert_eq!(response, "Echo: How are you?");

        // 3. 验证对话历史
        let context = agent.short_term_memory.get_context_messages(None); // 获取所有消息
        assert_eq!(context.len(), 4); // 2*(user + assistant)

        // 4. 测试上下文裁剪
        let trimmed = agent.short_term_memory.get_context_messages(Some(50));
        assert!(trimmed.len() <= context.len());

        // 5. 验证状态
        assert!(matches!(agent.state, AgentState::Ready));
    }

    #[tokio::test]
    async fn test_agent_tool_chain() {
        let agent = create_test_agent();

        // 1. 执行第一个工具
        let result1 = agent
            .execute_tool("echo", serde_json::json!({"text": "first call"}))
            .await
            .unwrap();

        // 2. 使用第一个工具的结果执行第二个工具
        if let ToolExecutionResult::Success { output, .. } = result1 {
            let result2 = agent
                .execute_tool("echo", serde_json::json!({"text": output}))
                .await
                .unwrap();

            // 3. 验证结果
            if let ToolExecutionResult::Success { output, .. } = result2 {
                assert_eq!(output, "first call");
            } else {
                panic!("Expected Success variant");
            }
        } else {
            panic!("Expected Success variant");
        }

        // 4. 验证工具调用历史
        let context = agent.short_term_memory.get_context_messages(None);
        let tool_messages = context
            .iter()
            .filter(|m| matches!(m.role, Role::Tool))
            .count();
        assert_eq!(tool_messages, 0); // 工具调用不会被添加到上下文中,因为我们直接调用了execute_tool
    }
}
