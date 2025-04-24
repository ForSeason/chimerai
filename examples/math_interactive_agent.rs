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
use std::io;
use tokio::io::{self as tokio_io, AsyncBufReadExt, AsyncWriteExt, BufReader};

#[tokio::main]
async fn main() -> Result<()> {
    math_interactive_agent().await?;
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

async fn math_interactive_agent() -> Result<()> {
    use std::env::var;

    let (api_key, model, api_url) = (var("API_KEY")?, var("MODEL")?, var("API_BASE_URL")?);

    let system_prompt = r##"
    你是一个数学助手，擅长解决各种数学问题。你可以使用calculator工具来进行计算。

    请遵循以下规则：
    1. 对于每一个计算步骤，清晰地展示你的推理过程
    2. 使用calculator工具进行实际的计算
    3. 不要跳步计算，必须基于已有的数字或者计算工具产生的中间结果
    4. 每次工具调用后，给出新的计算式，并基于这个计算式继续
    5. 确保最终结果是准确的
    
    举例：如果用户问 "13*17+19"，你应该：
    1. 使用calculator工具计算 13*17
    2. 根据工具返回的结果，再使用calculator工具计算这个结果+19
    3. 给出最终答案
    "##;

    let config = chimerai::types::AgentConfig {
        system_prompt: system_prompt.to_string(),
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

    // 交互式聊天逻辑
    let mut stdin = BufReader::new(tokio_io::stdin());
    let mut stdout = tokio_io::stdout();

    // 创建一个退出标志
    let mut exit_requested = false;

    println!("数学助手已启动，请输入您的数学问题...");
    println!("(输入\"\"\"开始多行输入模式)");

    while !exit_requested {
        // 提示用户输入
        stdout.write_all("\n> ".as_bytes()).await?;
        stdout.flush().await?;

        // 收集多行输入
        let mut user_message = String::new();
        let mut is_message_complete = false;
        let mut in_multiline_mode = false;

        while !is_message_complete {
            let mut buffer = String::new();
            match stdin.read_line(&mut buffer).await {
                Ok(0) => {
                    // EOF - 用户可能使用了Ctrl+D
                    stdout.write_all("\n检测到EOF，退出程序\n".as_bytes()).await?;
                    return Ok(());
                }
                Ok(_) => {
                    // 成功读取输入
                    let line = buffer.trim_end();
                    
                    // 检查是否是多行模式分隔符
                    if line == "\"\"\"" {
                        if in_multiline_mode {
                            // 结束多行模式
                            in_multiline_mode = false;
                            is_message_complete = true;
                        } else {
                            // 开始多行模式
                            in_multiline_mode = true;
                            // 不将分隔符本身添加到消息中
                        }
                    } else if in_multiline_mode {
                        // 在多行模式中，所有行（包括空行）都添加到消息中
                        if !user_message.is_empty() {
                            user_message.push('\n');
                        }
                        user_message.push_str(line);
                    } else {
                        // 非多行模式的处理
                        // 普通模式下，直接使用输入的内容（一行即可）
                        user_message = line.to_string();
                        is_message_complete = true;
                    }
                }
                Err(e) => {
                    stdout.write_all(format!("\n读取输入错误: {}\n", e).as_bytes()).await?;
                    return Err(anyhow!("读取输入错误: {}", e));
                }
            }
        }

        // 为用户显示输入已被接收
        stdout.write_all("\n正在计算中...\n".as_bytes()).await?;
        stdout.flush().await?;

        // 处理消息并获取响应
        let response_future = agent.handle_message(user_message);
        tokio::pin!(response_future);
        
        let timeout_duration = time::Duration::from_secs(600);
        let timeout_future = tokio::time::sleep(timeout_duration);
        tokio::pin!(timeout_future);
        
        tokio::select! {
            result = &mut response_future => {
                match result {
                    Ok(response) => {
                        // 输出响应
                        stdout.write_all(format!("\n{}\n", response).as_bytes()).await?;
                    },
                    Err(err) => {
                        // 输出错误
                        stdout.write_all(format!("\n计算出错: {}\n", err).as_bytes()).await?;
                    }
                }
            },
            _ = &mut timeout_future => {
                stdout.write_all("\n计算超时，请尝试简化问题重新提问\n".as_bytes()).await?;
            }
        }
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
        println!("计算工具调用: {args:?}");
        let op = args
            .get("op")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("缺少或无效的 'op' 参数"))?;

        let num1 = args
            .get("num1")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("缺少或无效的 'num1' 参数"))?;

        let num2 = args
            .get("num2")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("缺少或无效的 'num2' 参数"))?;

        let result = match op {
            "add" => num1 + num2,
            "subtract" => num1 - num2,
            "multiply" => num1 * num2,
            "divide" => {
                if num2 == 0.0 {
                    return Err(anyhow!("除数不能为零"));
                }
                num1 / num2
            }
            _ => return Err(anyhow!("不支持的操作: {}", op)),
        };

        Ok(format!("结果: {}", result))
    }
} 