use async_trait::async_trait;
use rig::{
    client::FinalCompletionResponse,
    completion::{self, CompletionError, CompletionRequest, CompletionResponse, GetTokenUsage},
    embeddings::{self, Embedding, EmbeddingError},
    streaming::StreamingCompletionResponse,
};
use std::sync::Arc;

#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    async fn embed_text(&self, input: &str) -> Result<Embedding, EmbeddingError>;
    async fn embed_texts(&self, input: Vec<String>) -> Result<Vec<Embedding>, EmbeddingError>;
    fn ndims(&self) -> usize;
}

#[async_trait]
impl<T> EmbeddingModel for T
where
    T: embeddings::EmbeddingModel + Send + Sync,
{
    async fn embed_text(&self, input: &str) -> Result<Embedding, EmbeddingError> {
        embeddings::EmbeddingModel::embed_text(self, input).await
    }

    async fn embed_texts(&self, input: Vec<String>) -> Result<Vec<Embedding>, EmbeddingError> {
        embeddings::EmbeddingModel::embed_texts(self, input).await
    }

    fn ndims(&self) -> usize {
        embeddings::EmbeddingModel::ndims(self)
    }
}

#[async_trait]
pub trait CompletionModel: Send + Sync {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<()>, CompletionError>;
}

#[derive(Clone)]
pub struct RigCompletionModelAdapter {
    inner: Arc<dyn CompletionModel>,
}

impl RigCompletionModelAdapter {
    pub fn new(inner: Arc<dyn CompletionModel>) -> Self {
        Self { inner }
    }
}

impl From<Box<dyn CompletionModel>> for RigCompletionModelAdapter {
    fn from(value: Box<dyn CompletionModel>) -> Self {
        Self {
            inner: Arc::from(value),
        }
    }
}

impl From<Arc<dyn CompletionModel>> for RigCompletionModelAdapter {
    fn from(value: Arc<dyn CompletionModel>) -> Self {
        Self { inner: value }
    }
}

impl completion::CompletionModel for RigCompletionModelAdapter {
    type Response = ();
    type StreamingResponse = FinalCompletionResponse;
    type Client = Arc<dyn CompletionModel>;

    fn make(client: &Self::Client, _model: impl Into<String>) -> Self {
        Self {
            inner: client.clone(),
        }
    }

    fn completion(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<Output = Result<CompletionResponse<Self::Response>, CompletionError>>
           + rig::wasm_compat::WasmCompatSend {
        let model = self.inner.clone();
        async move { model.completion(request).await }
    }

    fn stream(
        &self,
        _request: CompletionRequest,
    ) -> impl std::future::Future<
        Output = Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>,
    > + rig::wasm_compat::WasmCompatSend {
        async {
            Err(CompletionError::ResponseError(
                "Streaming is not supported by rig_dyn::CompletionModel adapter".to_string(),
            ))
        }
    }
}

#[async_trait]
impl<M> CompletionModel for M
where
    M: completion::CompletionModel + Send + Sync,
    M::StreamingResponse: Clone + Unpin + GetTokenUsage + 'static,
{
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<()>, CompletionError> {
        self.completion(request).await.map(|response| CompletionResponse {
            choice: response.choice,
            usage: response.usage,
            raw_response: (),
            message_id: response.message_id,
        })
    }
}
