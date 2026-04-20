use std::env;

use anyhow::Result;
use rig::{
    OneOrMany,
    completion::CompletionRequest,
    message::{self, Message},
};
use rig_dyn::Provider;

#[tokio::main]
async fn main() -> Result<()> {
    let provider = Provider::OpenAI;
    // get api key from somewhere
    let api_key = env::var("OPENAI_API_KEY").unwrap();
    let client = provider.client(&api_key, None)?;
    let completion_model = client.completion_model("gpt-4o").await;

    let request = CompletionRequest {
        model: None,
        additional_params: None,
        chat_history: OneOrMany::one(Message::user("Hello, World!")),
        documents: vec![],
        max_tokens: None,
        output_schema: None,
        preamble: Some("You are a helpful assistant.".to_string()),
        temperature: Some(0.7),
        tool_choice: None,
        tools: vec![],
    };

    let response = completion_model.completion(request).await?.first();

    if let message::AssistantContent::Text(content) = response {
        println!("{}", content.text);
    }

    Ok(())
}
