use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::UdpSocket;
use tokio::sync::Mutex;
use tokio::sync::mpsc::UnboundedSender;
use std::time::{SystemTime, Duration};
use crate::routing::{KBucketTable, PeerInfo, NodeId, InsertResult};
use crate::protocol::{Frame, FramePayload};

pub struct UdpTransport {
    pub self_id: NodeId,
    pub socket: Arc<UdpSocket>,
    pub table: Arc<Mutex<KBucketTable>>,
}

impl UdpTransport {
    pub async fn new(self_id: NodeId, addr: SocketAddr) -> Result<Self, std::io::Error> {
        let socket = UdpSocket::bind(addr).await?;
        let table = Arc::new(Mutex::new(KBucketTable::new(self_id)));
        Ok(Self {
            self_id,
            socket: Arc::new(socket),
            table,
        })
    }

    pub fn start(
        self: Arc<Self>,
        rx_forwarder: UnboundedSender<Frame>,
    ) {
        let transport = Arc::clone(&self);
        tokio::spawn(async move {
            let mut buf = [0u8; 1024];
            loop {
                tokio::select! {
                    res = transport.socket.recv_from(&mut buf) => {
                        match res {
                            Ok((size, src)) => {
                                if let Ok(frame) = Frame::parse(&buf[..size]) {
                                    if frame.sender == transport.self_id {
                                        continue;
                                    }
                                    
                                    // Handle peer routing table insertion
                                    let peer = PeerInfo {
                                        id: frame.sender,
                                        address: src,
                                    };
                                    
                                    let mut table_lock = transport.table.lock().await;
                                    let insert_res = table_lock.insert(peer);
                                    drop(table_lock);

                                    match insert_res {
                                        InsertResult::BucketFullPendingEviction { oldest, candidate } => {
                                            // Enforce LRU Eviction Guard
                                            let t_clone = Arc::clone(&transport);
                                            tokio::spawn(async move {
                                                // 1. Send Ping to oldest
                                                let ping_frame = Frame {
                                                    sender: t_clone.self_id,
                                                    payload: FramePayload::Ping {
                                                        timestamp: SystemTime::now()
                                                            .duration_since(SystemTime::UNIX_EPOCH)
                                                            .unwrap_or_default()
                                                            .as_millis() as u64,
                                                    },
                                                };
                                                let mut ping_buf = [0u8; 100];
                                                if let Ok(p_size) = ping_frame.serialize(&mut ping_buf) {
                                                    let _ = t_clone.socket.send_to(&ping_buf[..p_size], oldest.address).await;
                                                }

                                                // 2. Wait 500ms for Pong update
                                                tokio::time::sleep(Duration::from_millis(500)).await;

                                                // 3. Lock table and check if oldest is still at the front of the bucket
                                                let mut t_lock = t_clone.table.lock().await;
                                                let dist = t_clone.self_id.xor_distance(&candidate.id);
                                                let idx = t_lock.get_bucket_index(&dist);
                                                let bucket = &t_lock.buckets[idx];
                                                if bucket.peers[0] == Some(oldest) {
                                                    // Evict oldest peer and insert candidate
                                                    t_lock.evict_and_insert(oldest.id, candidate);
                                                    println!("[LRU Guard] Evicted inactive peer {:?} to insert candidate {:?}", oldest.id, candidate.id);
                                                } else {
                                                    println!("[LRU Guard] Oldest peer {:?} is active. Kept in bucket, candidate {:?} rejected.", oldest.id, candidate.id);
                                                }
                                            });
                                        }
                                        _ => {}
                                    }

                                    // Process frame payload
                                    match frame.payload {
                                        FramePayload::Ping { timestamp } => {
                                            // Auto-respond with Pong
                                            let pong = Frame {
                                                sender: transport.self_id,
                                                payload: FramePayload::Pong { timestamp },
                                            };
                                            let mut pong_buf = [0u8; 100];
                                            if let Ok(p_size) = pong.serialize(&mut pong_buf) {
                                                let _ = transport.socket.send_to(&pong_buf[..p_size], src).await;
                                            }
                                        }
                                        FramePayload::Pong { .. } => {
                                            // Handled automatically via K-bucket table insertion
                                        }
                                        FramePayload::ThermodynamicPush { .. } | FramePayload::GradientDiff { .. } => {
                                            // Forward to orchestrator channel
                                            let _ = rx_forwarder.send(frame);
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("[UDP Transport] Recv error: {:?}", e);
                            }
                        }
                    }
                }
            }
        });
    }

    pub async fn send_to(&self, frame: &Frame, addr: SocketAddr) -> Result<(), &'static str> {
        let mut buf = [0u8; 1024];
        let size = frame.serialize(&mut buf)?;
        let socket = Arc::clone(&self.socket);
        tokio::spawn(async move {
            let _ = socket.send_to(&buf[..size], addr).await;
        });
        Ok(())
    }
}
