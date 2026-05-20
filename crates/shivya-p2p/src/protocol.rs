use crate::routing::NodeId;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FramePayload {
    Ping { timestamp: u64 },
    Pong { timestamp: u64 },
    ThermodynamicPush { free_energy: f64, pressure: f64 },
    GradientDiff { target_id: NodeId, coefficient: f64, flow: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    pub sender: NodeId,
    pub payload: FramePayload,
}

impl Frame {
    /// Zero-heap reallocation binary frame decoder
    pub fn parse(buf: &[u8]) -> Result<Self, &'static str> {
        if buf.len() < 25 {
            return Err("Frame too short for header");
        }

        // Magic bytes validation
        if &buf[0..4] != &[0x53, 0x48, 0x56, 0x59] {
            return Err("Invalid frame magic sequence");
        }

        let frame_type = buf[4];

        let mut sender_bytes = [0u8; 20];
        sender_bytes.copy_from_slice(&buf[5..25]);
        let sender = NodeId(sender_bytes);

        let payload = match frame_type {
            0x01 => {
                if buf.len() < 33 {
                    return Err("Ping frame too short");
                }
                let mut ts_bytes = [0u8; 8];
                ts_bytes.copy_from_slice(&buf[25..33]);
                let timestamp = u64::from_be_bytes(ts_bytes);
                FramePayload::Ping { timestamp }
            }
            0x02 => {
                if buf.len() < 33 {
                    return Err("Pong frame too short");
                }
                let mut ts_bytes = [0u8; 8];
                ts_bytes.copy_from_slice(&buf[25..33]);
                let timestamp = u64::from_be_bytes(ts_bytes);
                FramePayload::Pong { timestamp }
            }
            0x03 => {
                if buf.len() < 41 {
                    return Err("ThermodynamicPush frame too short");
                }
                // Strict Big-Endian float decoding
                let mut fe_bytes = [0u8; 8];
                fe_bytes.copy_from_slice(&buf[25..33]);
                let free_energy = f64::from_bits(u64::from_be_bytes(fe_bytes));

                let mut pr_bytes = [0u8; 8];
                pr_bytes.copy_from_slice(&buf[33..41]);
                let pressure = f64::from_bits(u64::from_be_bytes(pr_bytes));

                FramePayload::ThermodynamicPush { free_energy, pressure }
            }
            0x04 => {
                if buf.len() < 61 {
                    return Err("GradientDiff frame too short");
                }
                let mut target_bytes = [0u8; 20];
                target_bytes.copy_from_slice(&buf[25..45]);
                let target_id = NodeId(target_bytes);

                // Strict Big-Endian float decoding
                let mut coef_bytes = [0u8; 8];
                coef_bytes.copy_from_slice(&buf[45..53]);
                let coefficient = f64::from_bits(u64::from_be_bytes(coef_bytes));

                let mut flow_bytes = [0u8; 8];
                flow_bytes.copy_from_slice(&buf[53..61]);
                let flow = f64::from_bits(u64::from_be_bytes(flow_bytes));

                FramePayload::GradientDiff { target_id, coefficient, flow }
            }
            _ => return Err("Unknown frame type action"),
        };

        Ok(Frame { sender, payload })
    }

    /// Zero-heap reallocation binary frame encoder
    pub fn serialize(&self, buf: &mut [u8]) -> Result<usize, &'static str> {
        if buf.len() < 25 {
            return Err("Buffer too small for header");
        }

        buf[0..4].copy_from_slice(&[0x53, 0x48, 0x56, 0x59]);
        buf[5..25].copy_from_slice(&self.sender.0);

        match self.payload {
            FramePayload::Ping { timestamp } => {
                buf[4] = 0x01;
                if buf.len() < 33 {
                    return Err("Buffer too small for Ping");
                }
                buf[25..33].copy_from_slice(&timestamp.to_be_bytes());
                Ok(33)
            }
            FramePayload::Pong { timestamp } => {
                buf[4] = 0x02;
                if buf.len() < 33 {
                    return Err("Buffer too small for Pong");
                }
                buf[25..33].copy_from_slice(&timestamp.to_be_bytes());
                Ok(33)
            }
            FramePayload::ThermodynamicPush { free_energy, pressure } => {
                buf[4] = 0x03;
                if buf.len() < 41 {
                    return Err("Buffer too small for ThermodynamicPush");
                }
                // Enforce strict Big-Endian float serialization
                buf[25..33].copy_from_slice(&free_energy.to_bits().to_be_bytes());
                buf[33..41].copy_from_slice(&pressure.to_bits().to_be_bytes());
                Ok(41)
            }
            FramePayload::GradientDiff { target_id, coefficient, flow } => {
                buf[4] = 0x04;
                if buf.len() < 61 {
                    return Err("Buffer too small for GradientDiff");
                }
                buf[25..45].copy_from_slice(&target_id.0);
                // Enforce strict Big-Endian float serialization
                buf[45..53].copy_from_slice(&coefficient.to_bits().to_be_bytes());
                buf[53..61].copy_from_slice(&flow.to_bits().to_be_bytes());
                Ok(61)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ping_pong_serialization() {
        let sender = NodeId([7u8; 20]);
        let frame = Frame {
            sender,
            payload: FramePayload::Ping { timestamp: 123456789 },
        };
        let mut buf = [0u8; 100];
        let size = frame.serialize(&mut buf).unwrap();
        assert_eq!(size, 33);

        let parsed = Frame::parse(&buf[..size]).unwrap();
        assert_eq!(parsed, frame);
    }

    #[test]
    fn test_thermodynamic_push_serialization() {
        let sender = NodeId([9u8; 20]);
        let frame = Frame {
            sender,
            payload: FramePayload::ThermodynamicPush {
                free_energy: -12.3456,
                pressure: 1.5,
            },
        };
        let mut buf = [0u8; 100];
        let size = frame.serialize(&mut buf).unwrap();
        assert_eq!(size, 41);

        let parsed = Frame::parse(&buf[..size]).unwrap();
        assert_eq!(parsed, frame);
    }
}
