#[allow(dead_code)] // public API exposed to embedders even though main.rs doesn't drive it directly
mod bridge;
mod telemetry;
mod orchestrator;

use clap::{Parser, Subcommand};
use std::sync::Arc;
use tokio::sync::Mutex;
use std::path::Path;
use tokio::net::UnixListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use telemetry::TelemetrySampler;
use orchestrator::NativeOrchestrator;
use std::net::SocketAddr;

use futures_util::StreamExt;
use futures_util::SinkExt;
use tokio_tungstenite::tungstenite::Message;

fn get_socket_path(port: u16) -> String {
    format!("/tmp/shivya_cli_{}.sock", port)
}

#[derive(Parser)]
#[command(name = "shivya-cli")]
#[command(about = "Shivya Headless Daemon CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the background daemon node
    Start {
        /// Spawn the orchestrator in the background
        #[arg(long)]
        daemon: bool,

        /// Port to bind the UDP listener on
        #[arg(long, default_value_t = 8085)]
        port: u16,

        /// Bootstrap peer socket address (e.g. 127.0.0.1:8085)
        #[arg(long)]
        peer: Option<SocketAddr>,

        /// Enable real-time WebSocket visualization server
        #[arg(long)]
        visualize: bool,
    },
    /// Query active memory segments and print metrics
    Status {
        /// Port of the target daemon to query
        #[arg(long, default_value_t = 8085)]
        port: u16,
    },
}

#[cfg(unix)]
async fn wait_for_signals() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut sigint = signal(SignalKind::interrupt()).expect("Failed to bind SIGINT listener");
    let mut sigterm = signal(SignalKind::terminate()).expect("Failed to bind SIGTERM listener");
    tokio::select! {
        _ = sigint.recv() => {
            println!("\n[Apoptosis] Received SIGINT. Running orderly apoptotic memory teardown...");
        }
        _ = sigterm.recv() => {
            println!("\n[Apoptosis] Received SIGTERM. Running orderly apoptotic memory teardown...");
        }
    }
}

#[cfg(not(unix))]
async fn wait_for_signals() {
    tokio::signal::ctrl_c().await.expect("Failed to bind Ctrl-C listener");
    println!("\n[Apoptosis] Received Ctrl-C. Running orderly apoptotic memory teardown...");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Cli::parse();

    match args.command {
        Commands::Start { daemon, port, peer, visualize } => {
            println!("Initializing Shivya 5-Layer Edge Daemon...");

            // Phase A: True OS-level fork must happen BEFORE the Tokio runtime
            // is constructed. Forking after the runtime starts would orphan I/O
            // descriptors held by reactor threads.
            #[cfg(unix)]
            {
                if daemon {
                    use daemonize::Daemonize;
                    use std::fs::File;

                    let pid_file = "/tmp/shivya.pid".to_string();
                    let out_file = File::create("/tmp/shivya.out")?;
                    let err_file = File::create("/tmp/shivya.err")?;

                    let d = Daemonize::new()
                        .pid_file(&pid_file)
                        .chown_pid_file(false)
                        .working_directory("/tmp")
                        .stdout(out_file)
                        .stderr(err_file);

                    match d.start() {
                        Ok(_) => {
                            // From this point we are the detached child;
                            // stdout/stderr now flow to /tmp/shivya.out/err.
                            eprintln!("[Daemon] Forked into background. PID file: {}", pid_file);
                        }
                        Err(e) => {
                            eprintln!("[Daemon] Fork failed: {}. Continuing in foreground.", e);
                        }
                    }
                }
            }

            // Build the multi-threaded Tokio runtime AFTER the fork so reactor
            // descriptors live in the detached child, not the parent shell.
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            runtime.block_on(run_start(port, peer, visualize))
        }
        Commands::Status { port } => {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            runtime.block_on(run_status(port))
        }
    }
}

async fn run_start(
    port: u16,
    peer: Option<SocketAddr>,
    visualize: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let socket_path = get_socket_path(port);

    if Path::new(&socket_path).exists() {
        println!("[UDS] Lingering stale socket file found. Performing clean-up.");
        let _ = std::fs::remove_file(&socket_path);
    }

    let self_id = shivya_p2p::routing::NodeId::random();
    let bind_addr: SocketAddr = format!("127.0.0.1:{}", port).parse()?;
    println!("[P2P] Starting UDP Transport on {} with Node ID {:?}", bind_addr, self_id);
    let transport = Arc::new(shivya_p2p::transport::UdpTransport::new(self_id, bind_addr).await?);
    let p2p_table = Arc::clone(&transport.table);
    let p2p_transport = Arc::clone(&transport);

    let mut orchestrator_inner = NativeOrchestrator::new(10);
    orchestrator_inner.set_p2p(self_id, p2p_table, p2p_transport);

    let orchestrator = Arc::new(Mutex::new(orchestrator_inner));
    let orchestrator_clone = Arc::clone(&orchestrator);

    let (tx_forwarder, mut rx_forwarder) = tokio::sync::mpsc::unbounded_channel();
    let transport_clone = Arc::clone(&transport);
    transport_clone.start(tx_forwarder);

    let orchestrator_p2p = Arc::clone(&orchestrator);
    tokio::spawn(async move {
        while let Some(frame) = rx_forwarder.recv().await {
            let mut orch = orchestrator_p2p.lock().await;
            orch.handle_p2p_frame(frame);
        }
    });

    // Bootstrap: PING + FIND_NODE(self) to seed iterative bucket discovery.
    if let Some(peer_addr) = peer {
        println!("[P2P] Bootstrap handshake → {}", peer_addr);
        let ping_frame = shivya_p2p::protocol::Frame {
            sender: self_id,
            payload: shivya_p2p::protocol::FramePayload::Ping {
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
            },
        };
        let _ = transport.send_to(&ping_frame, peer_addr).await;

        let find_self = shivya_p2p::protocol::Frame {
            sender: self_id,
            payload: shivya_p2p::protocol::FramePayload::FindNode { target: self_id },
        };
        let _ = transport.send_to(&find_self, peer_addr).await;
    }

    let ws_broadcast = if visualize {
        let (tx, _) = tokio::sync::broadcast::channel::<String>(100);
        let tx_clone = tx.clone();

        tokio::spawn(async move {
            let addr = "127.0.0.1:9002";
            if let Ok(listener) = tokio::net::TcpListener::bind(addr).await {
                println!("[Visualizer] WebSocket server listening on ws://{}", addr);
                while let Ok((stream, peer_addr)) = listener.accept().await {
                    let tx_rx = tx_clone.clone();
                    tokio::spawn(async move {
                        if let Ok(ws_stream) = tokio_tungstenite::accept_async(stream).await {
                            println!("[Visualizer] Client connected from {}", peer_addr);
                            let (mut ws_writer, _) = ws_stream.split();
                            let mut rx = tx_rx.subscribe();
                            loop {
                                match rx.recv().await {
                                    Ok(msg) => {
                                        if ws_writer.send(Message::Text(msg)).await.is_err() {
                                            break;
                                        }
                                    }
                                    Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                                        eprintln!("[Visualizer] Client lagging behind! Skipped {} messages.", skipped);
                                    }
                                    Err(_) => break,
                                }
                            }
                            println!("[Visualizer] Client disconnected: {}", peer_addr);
                        }
                    });
                }
            } else {
                eprintln!("[Visualizer] Failed to bind WebSocket server to {}", addr);
            }
        });
        Some(tx)
    } else {
        None
    };

    let ws_broadcast_clone = ws_broadcast.clone();
    tokio::spawn(async move {
        let mut sampler = TelemetrySampler::new();
        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(1000));
        loop {
            interval.tick().await;
            let snap = sampler.sample();
            let net_rate = (snap.net_rx + snap.net_tx) as f64;
            let mut orch = orchestrator_clone.lock().await;
            orch.step_with_telemetry(snap.cpu as f64, net_rate, snap.memory_used_ratio);
            if let Some(ref tx_chan) = ws_broadcast_clone {
                let status_json = orch.get_status_json();
                let _ = tx_chan.send(status_json);
            }
        }
    });

    let listener = UnixListener::bind(&socket_path)?;
    let orchestrator_uds = Arc::clone(&orchestrator);
    tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            let response = {
                let orch = orchestrator_uds.lock().await;
                orch.get_status_json()
            };
            let _ = stream.write_all(response.as_bytes()).await;
        }
    });

    println!("[Lifecycle] Node running. UDS listener bound to {}", socket_path);

    wait_for_signals().await;

    if Path::new(&socket_path).exists() {
        let _ = std::fs::remove_file(&socket_path);
    }
    println!("[Lifecycle] Apoptotic clean-up complete. Node gracefully terminated.");
    Ok(())
}

async fn run_status(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let socket_path = get_socket_path(port);
    if !Path::new(&socket_path).exists() {
        eprintln!("Error: Shivya daemon socket not found at {}. Is the daemon running?", socket_path);
        std::process::exit(1);
    }

    let mut stream = tokio::net::UnixStream::connect(&socket_path).await?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await?;

    let json_str = String::from_utf8_lossy(&response);

    match serde_json::from_str::<serde_json::Value>(&json_str) {
        Ok(status) => {
            println!("============================================================");
            println!("             SHIVYA HEADLESS DAEMON STATUS REGISTRY         ");
            println!("============================================================");
            println!("Collective Free Energy Level  : {:.4}", status["collective_free_energy"].as_f64().unwrap_or(0.0));
            println!("Topological Curl Deviation    : {:.4}", status["curl_deviation"].as_f64().unwrap_or(0.0));
            println!("Active Node Count             : {}", status["active_nodes_count"]);
            println!("Morphogenetic Active Pool     : {:?}", status["active_pool"]);
            println!("------------------------------------------------------------");
            println!("Node Memory Details:");
            if let Some(nodes) = status["nodes"].as_array() {
                for node in nodes {
                    if node["active"].as_bool().unwrap_or(false) {
                        println!("  Node #{} (ACTIVE):", node["id"]);
                        println!("    - Free Energy              : {:.4}", node["free_energy"].as_f64().unwrap_or(0.0));
                        println!("    - Belief Dimensions        : {}", node["belief_dim"]);
                        println!("    - Morphic Instruction Count: {} insts", node["instruction_count"]);
                        println!("    - Turing Morphogen (U / V) : {:.4} / {:.4}",
                            node["morphogen_u"].as_f64().unwrap_or(0.0),
                            node["morphogen_v"].as_f64().unwrap_or(0.0)
                        );
                        println!("    - Morphic AST Equation     : {}", node["morphic_equation"]);
                    }
                }
            }
            println!("============================================================");
        }
        Err(_) => {
            println!("{}", json_str);
        }
    }
    Ok(())
}
