use crate::types::{ToolCallArgs, ToolCalls};
use crate::{llm::LLMClient, Decision, Message, Tool};
use anyhow::*;
use async_trait::async_trait;
use futures::{Stream, StreamExt, TryStreamExt};
use reqwest::Client;
use serde_json::json;
use std::collections::HashMap;
use std::pin::Pin;
use std::result::Result::Ok;
use tracing::debug;

pub struct OpenaiLlmClient {
    pub api_key: String,
    pub model: String,
    /// 例如：https://api.openai.com/v1/chat/completions
    pub api_url: String,
    /// 可选的超时设置等
    pub client: Client,
}

#[async_trait]
impl LLMClient for OpenaiLlmClient {
    async fn complete(
        &self,
        messages: &[Message],
        tools: Vec<&Box<dyn Tool>>,
        max_tokens: Option<usize>,
    ) -> Result<Decision> {
        // 1. 转换 messages 为 OpenAI 格式
        let openai_messages = convert_messages(messages);

        // 2. 转换 tools 为 OpenAI function 格式
        let openai_functions = convert_tools_to_openai_functions(&tools);

        // 3. 构造请求体
        let mut request_body = serde_json::json!({
            "model": self.model,
            "messages": openai_messages,
            "tools": openai_functions,
            "tool_choice": "auto",
            "temperature": 0.7,
            "stream": false
        });

        if let Some(max) = max_tokens {
            request_body["max_tokens"] = serde_json::json!(max);
        }

        debug!("request: {}", request_body.to_string());

        // 4. 发送请求
        let response = self
            .client
            .post(&self.api_url)
            .header("Content-Type", "application/json")
            .bearer_auth(&self.api_key)
            .json(&request_body)
            .send()
            .await?;

        let code = response.status();
        let response_text = response.text().await?.to_string();
        debug!("response: {code:?} {response_text}");
        let response_json: serde_json::Value = serde_json::from_str(&response_text)?;

        // 5. 解析响应
        parse_openai_response_into_decision(response_json)
    }

    async fn stream_complete(
        &self,
        messages: &[Message],
        tools: Vec<&Box<dyn Tool>>,
        max_tokens: Option<usize>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Decision>> + Send>>> {
        // 1. 将 messages 与 tools 转换为 OpenAI 所需格式
        let openai_messages = convert_messages(messages);
        let openai_functions = convert_tools_to_openai_functions(&tools);

        // 2. 构造请求体，注意 stream 字段设为 true
        let mut request_body = serde_json::json!({
            "model": self.model,
            "messages": openai_messages,
            "tools": openai_functions,
            "tool_choice": "auto",
            "temperature": 0.7,
            "stream": true,
        });
        if let Some(max) = max_tokens {
            request_body["max_tokens"] = serde_json::json!(max);
        }
        debug!("stream request: {}", request_body.to_string());

        // 3. 发送请求
        let response = self
            .client
            .post(&self.api_url)
            .header("Content-Type", "application/json")
            .bearer_auth(&self.api_key)
            .json(&request_body)
            .send()
            .await?;
        debug!("stream status: {}", response.status());

        // 4. 获取响应字节流
        let byte_stream = response.bytes_stream();

        // 将每个字节块转换为字符串，并按行拆分，过滤掉不需要的部分（例如 "[DONE]"）
        // 假设 byte_stream 的类型为 impl Stream<Item = Result<bytes::Bytes, reqwest::Error>>
        let line_stream = byte_stream
            .map_err(|e: reqwest::Error| anyhow!(e))
            .and_then(|chunk| async move {
                // 使用 chunk.as_ref() 来获取 &[u8]
                let s = std::str::from_utf8(chunk.as_ref())
                    .map_err(|e| anyhow!("UTF8 error: {}", e))?
                    .to_string();
                Ok(s)
            })
            .map_ok(|chunk_str| {
                // 将 chunk_str 中的行过滤并收集到 Vec<String> 中，保证每个 String 是独立拥有的
                let vec: Vec<String> = chunk_str
                    .lines()
                    .filter_map(|line| {
                        let trimmed = line.trim();
                        if trimmed.starts_with("data:") {
                            let data = trimmed.trim_start_matches("data:").trim();
                            if data.is_empty() {
                                None
                            } else if data == "[DONE]" {
                                println!("");
                                None
                            } else {
                                Some(data.to_string())
                            }
                        } else {
                            None
                        }
                    })
                    .collect();
                // 将 Vec 转换为 stream，注意这里迭代器中的每个 String 都是 owned 的
                futures::stream::iter(vec.into_iter().map(Ok::<String, anyhow::Error>))
            })
            .try_flatten();

        // 5. 将每一行的 JSON 字符串转换为 Decision（调用辅助函数解析每个流式 chunk）
        let decision_stream = line_stream.map(|json_line_result: Result<String>| {
            // 解析每一行 JSON，生成 Decision
            let json_line = json_line_result?;
            debug!("stream recieved: {json_line}");
            let json_value: serde_json::Value =
                serde_json::from_str(&json_line).map_err(|e| anyhow!("JSON parse error: {}", e))?;
            parse_openai_stream_chunk_into_decision(json_value)
        });

        Ok(Box::pin(decision_stream))
    }
}

/// 将 `Vec<Message>` 转换为 OpenAI 的 `messages`
fn convert_messages(messages: &[Message]) -> Vec<serde_json::Value> {
    messages
        .iter()
        .filter_map(|m| {
            match m {
                Message::Developer { content } => {
                    // 有些人会将 Developer 也当作 "system" 角色
                    Some(json_msg("system", content, None, None))
                }
                Message::System { content } => Some(json_msg("system", content, None, None)),
                Message::User { content } => Some(json_msg("user", content, None, None)),
                Message::Assistant {
                    content,
                    tool_calls,
                } => {
                    // 如果是单纯的 assistant 输出，则 content 直接放入
                    Some(json_msg("assistant", content, None, tool_calls.clone()))
                }
                Message::Tool {
                    content,
                    tool_call_id,
                } => {
                    // 工具调用的响应需要包含 tool_call_id
                    Some(serde_json::json!({
                        "role": "tool",
                        "content": content,
                        "tool_call_id": tool_call_id
                    }))
                }
            }
        })
        .collect()
}

/// 组装为 {"role": ..., "content": ...} 格式
fn json_msg(
    role: &str,
    content: &str,
    name: Option<&str>,
    tool_calls: Option<ToolCalls>,
) -> serde_json::Value {
    let mut res = serde_json::json!({
        "role": role,
        "content": content,
    });
    if let Some(name) = name {
        res["name"] = name.into();
    }
    if let Some(tool_calls) = tool_calls {
        res["tool_calls"] = tool_calls
            .iter()
            .map(|(tool_call_id, args)| {
                json!({
                    "id": tool_call_id,
                    "type": args.tool_type,
                    "function": {
                        "arguments": args.args.to_string(),
                        "name": args.tool_name,
                    },
                })
            })
            .collect::<Vec<_>>()
            .into();
    }
    res
}

/// 将本地的 `Tool` 转换为 OpenAI Functions 定义
fn convert_tools_to_openai_functions(tools: &[&Box<dyn Tool>]) -> Vec<serde_json::Value> {
    tools
        .iter()
        .map(|tool| {
            let mut function = serde_json::json!({
                "type": "function",
                "function": {
                    "name": tool.name(),
                }
            });
            if let Some(description) = tool.description() {
                function["function"]["description"] = description.into();
            }
            if let Some(args) = tool.args_schema() {
                function["function"]["parameters"] = args.clone();
            }
            function
        })
        .collect()
}

/// 解析OpenAI返回的JSON，根据是否有function_call来决定返回ExecuteTool或Respond
fn parse_openai_response_into_decision(response_json: serde_json::Value) -> Result<Decision> {
    let empty = vec![];
    let choices = response_json["choices"].as_array().unwrap_or(&empty);
    if choices.is_empty() {
        // 没有choices就返回一个空响应
        return Ok(Decision::Respond("".to_string()));
    }
    let message = &choices[0]["message"];
    let content = message["content"].as_str().unwrap_or("").to_string();

    // 检查是否有工具调用
    if let Some(tool_calls) = message["tool_calls"].as_array() {
        let mut tool_calls_map = HashMap::new();

        for tool_call in tool_calls {
            if let (Some(id), Some(function)) =
                (tool_call["id"].as_str(), tool_call["function"].as_object())
            {
                if let (Some(name), Some(args_str)) =
                    (function["name"].as_str(), function["arguments"].as_str())
                {
                    let parsed_args = match serde_json::from_str(args_str) {
                        Ok(v) => v,
                        Err(_) => serde_json::json!({}),
                    };

                    tool_calls_map.insert(
                        id.to_string(),
                        ToolCallArgs {
                            tool_type: "function".to_string(),
                            tool_name: name.to_string(),
                            args: parsed_args,
                        },
                    );
                }
            }
        }

        if !tool_calls_map.is_empty() {
            if !content.is_empty() {
                eprintln!("{content}");
            }
            return Ok(Decision::ExecuteTool(content, tool_calls_map));
        }
    }

    // 如果没有工具调用或工具调用解析失败，返回内容
    Ok(Decision::Respond(content))
}

/// 将流式返回的 JSON chunk 解析为 Decision。
/// 该函数根据 chunk 中 "choices" 内的 "delta" 字段提取 assistant 的内容或工具调用信息。
fn parse_openai_stream_chunk_into_decision(chunk: serde_json::Value) -> Result<Decision> {
    // 流式返回的 chunk 结构类似：
    // {
    //   "choices": [
    //     {
    //       "delta": { "content": "部分内容", "tool_calls": [...] },
    //       "index": 0,
    //       "finish_reason": null
    //     }
    //   ]
    // }
    let choices = match chunk["choices"].as_array() {
        Some(c) => c,
        None => return Ok(Decision::Respond(String::new())),
    };
    let delta = &choices[0]["delta"];
    let content = delta["content"].as_str().unwrap_or("").to_string();

    // 如果有 tool_calls，则构造 ExecuteTool 决策
    if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
        let mut tool_calls_map = std::collections::HashMap::new();
        for tool_call in tool_calls {
            if let (Some(id), Some(function)) = (
                tool_call.get("id").and_then(|v| v.as_str()),
                tool_call.get("function").and_then(|v| v.as_object()),
            ) {
                if let (Some(name), Some(args_str)) = (
                    function.get("name").and_then(|v| v.as_str()),
                    function.get("arguments").and_then(|v| v.as_str()),
                ) {
                    let parsed_args =
                        serde_json::from_str(args_str).unwrap_or(serde_json::json!({}));
                    tool_calls_map.insert(
                        id.to_string(),
                        ToolCallArgs {
                            tool_type: "function".to_string(),
                            tool_name: name.to_string(),
                            args: parsed_args,
                        },
                    );
                }
            }
        }
        if !tool_calls_map.is_empty() {
            // 如果同时有 content 和 tool_calls，可以选择先输出部分内容
            if !content.is_empty() {
                eprintln!("Partial content: {}", content);
            }
            return Ok(Decision::ExecuteTool(content, tool_calls_map));
        }
    }
    Ok(Decision::Respond(content))
}
