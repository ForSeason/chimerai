use anyhow::{anyhow, Result};
use std::collections::HashMap;
use tokio::time::timeout;

use crate::{
    llm::LLMClient,
    memory::{LongTermMemory, ShortTermMemory},
    tools::Tool,
    types::{AgentConfig, AgentState, Decision, Message, ToolCallArgs, ToolExecutionResult},
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
        self.short_term_memory.add_message(Message::System {
            content: config.system_prompt.clone(),
        });
        self.config = config;
        self
    }

    pub fn register_tool<T: Tool + 'static>(&mut self, tool: T) {
        self.tools.insert(tool.name(), Box::new(tool));
    }

    /// 处理传入的消息，并根据消息内容进行相应的操作
    ///
    /// 1. 检查代理当前状态是否为Ready，如果不是则返回错误
    /// 2. 将状态设置为Processing，表示正在处理消息
    /// 3. 将用户的消息添加到短期记忆中
    /// 4. 获取裁剪后的上下文消息，确保不超过最大token数
    /// 5. 进入循环，最多重试max_retries次：
    ///     a. 调用get_decision获取LLM的决策结果，并设置超时时间
    ///     b. 处理决策结果：
    ///         - 如果需要执行工具：
    ///             * 将助手的回应和工具调用信息添加到短期记忆中
    ///             * 执行所有指定的工具
    ///             * 根据工具执行结果，更新短期记忆中的内容
    ///         - 如果直接回应用户：
    ///             * 将助手的消息添加到短期记忆中
    ///             * 恢复代理状态为Ready
    ///             * 返回响应消息
    ///     c. 超时处理：增加重试次数或返回错误
    /// 6. 循环结束后，如果超过重试次数则返回相应错误
    pub async fn handle_message(&mut self, message: String) -> Result<String> {
        // 1. 状态检查
        if !matches!(self.state, AgentState::Ready) {
            return Err(anyhow!("Agent is not in ready state"));
        }
        self.state = AgentState::Processing;

        // 2. 添加用户消息到短期记忆
        self.short_term_memory
            .add_message(Message::User { content: message });

        // 3. 获取裁剪后的上下文
        let mut context = self
            .short_term_memory
            .get_context_messages(self.config.max_tokens);

        // 4. 循环处理直到得到最终响应
        let mut retries = 0;
        while retries < self.config.retry_config.max_retries {
            // 设置超时
            match timeout(self.config.timeout, self.get_decision(&context)).await {
                Ok(decision_result) => {
                    let decision = decision_result?;
                    match decision {
                        Decision::ExecuteTool(respond, tool_calls) => {
                            self.short_term_memory.add_message(Message::Assistant {
                                content: respond.clone(),
                                tool_calls: Some(tool_calls.clone()),
                            });
                            let ToolExecutionResult {
                                success_result,
                                failure_result,
                            } = self.execute_tool(&tool_calls).await?;
                            success_result
                                .into_iter()
                                .for_each(|(tool_call_id, content)| {
                                    self.short_term_memory.add_message(Message::Tool {
                                        content,
                                        tool_call_id,
                                    });
                                });
                            failure_result.into_iter().for_each(
                                        |(tool_call_id, error)| {
                                            self.short_term_memory.add_message(Message::Tool {
                                                content: format!(
                                                    "工具 {} 执行失败（错误信息：{}）。由于无法重试，请考虑使用其他方式解决问题或给出合适的响应。",
                                                    tool_calls.get(&tool_call_id).unwrap().tool_name,
                                                    error,
                                                ),
                                                tool_call_id,
                                            });
                                        },
                                    );
                            context = self
                                .short_term_memory
                                .get_context_messages(self.config.max_tokens);
                            continue;
                        }
                        Decision::Respond(response) => {
                            self.short_term_memory.add_message(Message::Assistant {
                                content: response.clone(),
                                tool_calls: None,
                            });
                            self.state = AgentState::Ready;
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
        let tools: Vec<&Box<dyn Tool>> = self.tools.values().collect();

        self.llm
            .complete(messages, tools, self.config.max_tokens)
            .await
    }

    /// 执行一系列工具调用，并收集它们的结果。
    ///
    /// 该函数接收一组工具调用请求，每个请求包含工具名称及其相关参数。对每个工具进行执行后，将结果存储在一个哈希映射中，其中键为工具名称，值为执行结果。如果任何一个工具调用失败，整个函数返回错误信息。
    ///
    /// # 参数
    /// * `tool_calls` - 一个包含多个`ToolCall`对象的向量，每个对象表示一次待执行的工具调用及其参数。
    ///
    /// # 返回值
    /// 如果所有工具成功执行，则返回一个`Result<HashMap<String, String>>`，其中键为工具名称，值为相应的执行结果。如果任何工具调用失败，则返回包含错误信息的`Result::Err`。
    async fn execute_tool(
        &self,
        args: &HashMap<String, ToolCallArgs>,
    ) -> Result<ToolExecutionResult> {
        let mut success_result: HashMap<String, String> = HashMap::new();
        let mut failure_result: HashMap<String, String> = HashMap::new();
        let tools = args
            .iter()
            .filter_map(|(tool_call_id, args)| {
                let tool = self.tools.get(&args.tool_name);
                if let None = tool {
                    failure_result.insert(
                        args.tool_name.clone(),
                        format!("Tool {} does not exist!", args.tool_name),
                    );
                    None
                } else {
                    Some((tool.unwrap(), &args.args, tool_call_id))
                }
            })
            .collect::<Vec<_>>();
        for (tool, args, tool_call_id) in tools {
            match tool.execute(args.clone()).await {
                Ok(result) => {
                    success_result.insert(tool_call_id.clone(), result);
                }
                Err(err) => {
                    failure_result.insert(tool_call_id.clone(), err.to_string());
                }
            }
        }

        Ok(ToolExecutionResult {
            success_result,
            failure_result,
        })
    }

    // async fn store_tool_result(&mut self, result: &str, tool_name: &str) -> Result<()> {
    //     let memory_entry = MemoryEntry {
    //         result: result.into(),
    //         metadata: MemoryMetadata {
    //             timestamp: chrono::Utc::now(),
    //             tags: vec!["tool_result".to_string(), tool_name.to_string()],
    //             source: tool_name.to_string(),
    //         },
    //     };

    //     self.long_term_memory.store(memory_entry).await
    // }
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
    use serde_json::json;
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
            max_tokens: Some(1000),
            enable_parallel: false,
            retry_config: crate::types::RetryConfig {
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
        let context = agent.short_term_memory.get_context_messages(None);
        assert_eq!(context.len(), 3); // system message + user message + assistant response
        assert_eq!(
            context[0],
            Message::System {
                content: "You are a helpful assistant.".to_string()
            },
        );
        assert_eq!(
            context[1],
            Message::User {
                content: "Hello".to_string()
            },
        );
        assert_eq!(
            context[2],
            Message::Assistant {
                content: "Echo: Hello".to_string(),
                tool_calls: None,
            },
        );
    }

    #[tokio::test]
    async fn test_agent_tool_execution() {
        let agent = create_test_agent();
        let tool_call_id = "tool_call_id".to_string();
        let mut args = HashMap::new();
        args.insert(
            tool_call_id.clone(),
            ToolCallArgs {
                tool_type: "function".to_string(),
                tool_name: "echo".to_string(),
                args: json!({"text": "test message"}),
            },
        );

        // 测试工具执行
        let result = agent
            .execute_tool(&args)
            // .execute_tool("echo", serde_json::json!({"text": "test message"}))
            .await
            .unwrap();

        assert_eq!(result.failure_result.len(), 0);
        assert_eq!(result.success_result.len(), 1);
        assert_eq!(
            *result.success_result.get(&tool_call_id).unwrap(),
            "test message"
        );
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
        let context = agent.short_term_memory.get_context_messages(None);
        assert_eq!(context.len(), 5); // system + 2*(user + assistant)

        // 3. 验证最近的对话
        let recent_messages = agent.short_term_memory.get_context_messages(None);
        assert!(!recent_messages.is_empty());
        assert_eq!(
            *recent_messages.last().unwrap(),
            Message::Assistant {
                content: "Echo: Second message".into(),
                tool_calls: None,
            },
        );

        // 4. 测试长期记忆存储和检索
        // let test_data = serde_json::json!({
        //     "important_info": "test_data"
        // });

        // let memory_entry = MemoryEntry {
        //     result: test_data.clone(),
        //     metadata: MemoryMetadata {
        //         timestamp: chrono::Utc::now(),
        //         tags: vec!["test".to_string()],
        //         source: "test_tool".to_string(),
        //     },
        // };

        // agent.long_term_memory.store(memory_entry).await.unwrap();

        // // 5. 通过语义查询验证记忆
        // let results = agent
        //     .long_term_memory
        //     .recall(&MemoryQuery::Semantic {
        //         description: "test data".to_string(),
        //         limit: 1,
        //     })
        //     .await
        //     .unwrap();

        // assert_eq!(results.len(), 1);
        // assert_eq!(results[0].content, test_data);
    }

    #[tokio::test]
    async fn test_agent_error_handling() {
        let mut agent = create_test_agent();

        // 1. 测试无效的工具调用
        let mut args1 = HashMap::new();
        args1.insert(
            "id".into(),
            ToolCallArgs {
                tool_type: "function".into(),
                tool_name: "what_tool".into(),
                args: json!({}),
            },
        );
        let result = agent.execute_tool(&args1).await;
        assert!(!result.unwrap().failure_result.is_empty());

        // 2. 测试参数缺失的工具调用
        let mut args2 = HashMap::new();
        args2.insert(
            "id".into(),
            ToolCallArgs {
                tool_type: "function".into(),
                tool_name: "echo".into(),
                args: json!({}),
            },
        );
        let result = agent.execute_tool(&args2).await;
        assert!(!result.unwrap().failure_result.is_empty());

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
        assert_eq!(context.len(), 5); // system + 2*(user + assistant)

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
        let mut args = HashMap::new();
        args.insert(
            "id1".into(),
            ToolCallArgs {
                tool_type: "function".into(),
                tool_name: "echo".into(),
                args: json!({"text": "first call"}),
            },
        );
        let result1 = agent.execute_tool(&args).await.unwrap();
        assert_eq!(result1.failure_result.is_empty(), true);
        assert_eq!(result1.success_result.len(), 1);
        // 2. 使用第一个工具的结果执行第二个工具
        let (_, output) = result1.success_result.iter().next().unwrap();
        args.insert(
            "id1".into(),
            ToolCallArgs {
                tool_type: "function".into(),
                tool_name: "echo".into(),
                args: json!({"text": output}),
            },
        );
        let result2 = agent.execute_tool(&args).await.unwrap();
        assert_eq!(result2.failure_result.is_empty(), true);
        assert_eq!(result2.success_result.len(), 1);

        // 4. 验证工具调用历史
        let context = agent.short_term_memory.get_context_messages(None);
        let tool_messages = context
            .iter()
            .filter(|m| matches!(m, Message::Tool { .. }))
            .count();
        assert_eq!(tool_messages, 0); // 工具调用不会被添加到上下文中,因为我们直接调用了execute_tool
    }
}
