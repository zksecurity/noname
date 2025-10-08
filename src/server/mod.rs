#[cfg(feature = "server")]
use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::get,
    Json, Router,
};
use base64::prelude::*;
use serde::Serialize;
use std::sync::Arc;
use std::thread;
use tokio::sync::{mpsc, Mutex};
#[cfg(feature = "server")]
use tower_http::services::ServeDir;

use crate::cli::packages::path_to_release_dir;

//
// The interface for the rest of the compiler which doesn't use async/await
//

#[derive(Clone)]
pub enum ServerMessage {
    Resume,
}

#[derive(Clone, Debug)]
pub enum CompilerMessage {
    State { title: String, state: String },
}

pub struct ServerShim {
    tx: mpsc::Sender<CompilerMessage>,
    rx: mpsc::Receiver<ServerMessage>,
}

impl ServerShim {
    #[cfg(feature = "server")]
    pub(crate) fn start_server() -> (thread::JoinHandle<()>, Self) {
        let (compiler_to_server, rx) = mpsc::channel(1024);
        let (tx, compiler_from_server) = mpsc::channel(1024);

        // then run the server in a thread
        let handle = thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(run_server(tx, rx));
        });

        let server_shim = ServerShim {
            tx: compiler_to_server,
            rx: compiler_from_server,
        };

        (handle, server_shim)
    }

    pub(crate) fn send<T: Serialize>(&self, title: String, state: &T) {
        #[cfg(feature = "server")]
        {
            let state = serde_json::to_string(&state).unwrap();
            let state = BASE64_STANDARD.encode(state);
            let _ = self
                .tx
                .clone()
                .try_send(CompilerMessage::State { title, state });
        }
    }

    pub(crate) fn recv(&mut self) -> Option<ServerMessage> {
        #[cfg(feature = "server")]
        {
            return Some(ServerMessage::Resume);
            //self.rx.blocking_recv()
        }
        #[cfg(not(feature = "server"))]
        {
            None
        }
    }
}

//
// Server
//

#[derive(Debug, Clone)]
struct ServerState {
    tx: Arc<Mutex<mpsc::Sender<ServerMessage>>>,
    compiler_state: Arc<Mutex<Vec<CompilerStep>>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompilerStep {
    pub title: String,
    pub state: String,
}

#[cfg(feature = "server")]
async fn run_server(tx: mpsc::Sender<ServerMessage>, rx: mpsc::Receiver<CompilerMessage>) {
    // initialize tracing
    tracing_subscriber::fmt::init();

    // create a new state
    let compiler_state = Arc::new(Mutex::new(Vec::<CompilerStep>::new()));
    let tx = Arc::new(Mutex::new(tx));
    let rx = Arc::new(Mutex::new(rx));
    let server_state = ServerState {
        tx,
        compiler_state: compiler_state.clone(),
    };

    tokio::spawn(async move {
        while let Some(msg) = rx.lock().await.recv().await {
            match msg {
                CompilerMessage::State { title, state } => {
                    let mut compiler_state = compiler_state.lock().await;
                    compiler_state.push(CompilerStep { title, state });
                }
            }
        }
    });

    // build our application with a route
    // let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    // load manifest dir from the environment, otherwise use the release directory
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| path_to_release_dir().to_string());
    let assets_dir = std::path::Path::new(&manifest_dir).join("assets");

    let app = Router::new()
        // let the compiler go to the next step
        .route("/states", get(states))
        .route("/resume/:id", get(resume))
        // get the state at the given counter
        .route("/state/:counter", get(get_state))
        .nest_service("/", ServeDir::new(assets_dir))
        .with_state(server_state);

    // run our app with hyper, listening globally on port 3000
    println!("listening on http://0.0.0.0:3000");
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

//
// Routes
//

#[cfg(feature = "server")]
#[axum::debug_handler]
async fn states(state: State<ServerState>) -> String {
    let compiler_state = state.compiler_state.lock().await;
    format!("{}", compiler_state.len())
}

#[cfg(feature = "server")]
#[axum::debug_handler]
async fn resume(
    Path(id): Path<usize>,
    state: State<ServerState>,
) -> Result<String, (StatusCode, String)> {
    let len = { state.compiler_state.lock().await.len() };
    if len != id {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("Invalid id: {} (expected: {})", id, len),
        ));
    }
    match state.tx.lock().await.send(ServerMessage::Resume).await {
        Ok(_) => Ok("Resuming...".to_string()),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to send message: {}", e),
        )),
    }
}

#[derive(Clone, Debug, Serialize)]
struct Response {
    message: CompilerStep,
}

#[cfg(feature = "server")]
#[axum::debug_handler]
async fn get_state(
    Path(counter): Path<usize>,
    state: State<ServerState>,
) -> Result<Json<Response>, (StatusCode, String)> {
    let compiler_state = state.compiler_state.lock().await;

    if let Some(msg) = compiler_state.get(counter) {
        let response = Response {
            message: msg.clone(),
        };
        Ok(Json(response))
    } else {
        Err((StatusCode::NOT_FOUND, "State not found".to_string()))
    }
}

//
// Tests
//

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // disabled for now because it hangs in test runner
    #[cfg(feature = "server")]
    fn test_server() {
        let (handle, _) = ServerShim::start_server();
        // wait on handle
        handle.join().unwrap();
    }
}
