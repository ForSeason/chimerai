use std::time;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use chimerai::llm::openai::OpenaiLlmClient;
use chimerai::Tool;
use chimerai::{
    memory::{MemoryEntry, MemoryQuery},
    LongTermMemory, Message, ShortTermMemory,
};
use serde_json::Value;
#[tokio::main]
async fn main() -> Result<()> {
    math_agent().await?;
    Ok(())
}

struct LTM {}

struct STM {
    messages: Vec<Message>,
}

#[async_trait]
impl LongTermMemory for LTM {
    // 存储记忆
    async fn store(&mut self, _entry: MemoryEntry) -> Result<()> {
        Ok(())
    }

    // 检索记忆
    async fn recall(&self, _query: &MemoryQuery) -> Result<Vec<MemoryEntry>> {
        Ok(vec![])
    }

    // 删除记忆
    async fn forget(&mut self, _query: &MemoryQuery) -> Result<()> {
        Ok(())
    }
}

#[async_trait]
impl ShortTermMemory for STM {
    /// 添加一条消息到短期记忆
    fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// 获取当前的对话上下文，根据 token 限制进行裁剪
    /// 如果 max_tokens 为 None，则返回所有消息
    fn get_context_messages(&self, _max_tokens: Option<usize>) -> Vec<Message> {
        self.messages.clone()
    }
}

async fn math_agent() -> Result<()> {
    use std::env::var;

    let (api_key, model, api_url) = (var("API_KEY")?, var("MODEL")?, var("API_BASE_URL")?);

    let config = chimerai::types::AgentConfig {
        system_prompt: String::new(),
        max_turns: 50,
        max_tokens: None,
        enable_parallel: true,
        retry_config: chimerai::types::RetryConfig {
            max_retries: 1,
            retry_delay: time::Duration::new(0, 100),
            should_retry_on_error: false,
        },
        temperature: 0.7,
        timeout: time::Duration::from_secs(600),
    };
    let long_term_memory = LTM {};
    let short_term_memory = STM { messages: vec![] };
    let llm = OpenaiLlmClient {
        api_key,
        model,
        api_url,
        client: reqwest::Client::new(),
    };
    let mut agent =
        chimerai::Agent::new(long_term_memory, short_term_memory, llm).with_config(config);

    agent.register_tool(CalcTool::new());

    let question = r##"
    使用提供的计算工具，回答给定问题。注意不要跳步计算，你的计算必须基于已有的数字或者计算工具产生的中间结果。每次工具调用后，你都需要给出新的计算式，并基于这个计算式继续调用工具。

    以下是问题：
    298345+238*2357*(44/11-2) = ?
    "##;

    println!("question:\n{question}\n\n==========\n");

    match agent.handle_message(question.to_string()).await {
        Ok(s) => println!("{s}"),
        Err(e) => println!("error: {e:?}"),
    }

    Ok(())
}

#[derive(Debug, Clone)]
pub struct CalcTool;

impl CalcTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for CalcTool {
    fn name(&self) -> String {
        "calculator".to_string()
    }

    fn description(&self) -> Option<String> {
        Some("A versatile calculator tool that supports addition, subtraction, multiplication and division".to_string())
    }

    fn args_schema(&self) -> Option<Value> {
        Some(serde_json::json!({
            "type": "object",
            "properties": {
                "op": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "Operation to perform"
                },
                "num1": {
                    "type": "number",
                    "description": "First operand"
                },
                "num2": {
                    "type": "number",
                    "description": "Second operand"
                }
            },
            "required": ["op", "num1", "num2"]
        }))
    }

    async fn execute(&self, args: Value) -> Result<String> {
        println!("tool called: {args:?}");
        let op = args
            .get("op")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("Missing or invalid 'op' argument"))?;

        let num1 = args
            .get("num1")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("Missing or invalid 'num1' argument"))?;

        let num2 = args
            .get("num2")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("Missing or invalid 'num2' argument"))?;

        let result = match op {
            "add" => num1 + num2,
            "subtract" => num1 - num2,
            "multiply" => num1 * num2,
            "divide" => {
                if num2 == 0.0 {
                    return Err(anyhow!("Division by zero"));
                }
                num1 / num2
            }
            _ => return Err(anyhow!("Unsupported operation: {}", op)),
        };

        Ok(format!("result: {:.2}", result)) // 保留两位小数
    }
}
