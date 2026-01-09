# Vivint WebRTC Two-Way Audio Protocol

This document describes the complete WebRTC signaling protocol used by Vivint cameras for two-way audio, reverse-engineered from the Vivint Android app.

## Overview

Vivint uses **Firebase Realtime Database** for WebRTC signaling (NOT gRPC or PubNub as initially suspected). The signaling exchanges SDP offers/answers and ICE candidates via Firebase database paths.

## Architecture

```
┌─────────────────┐     Firebase Realtime DB      ┌─────────────────┐
│  Vivint Mobile  │◄────────────────────────────►│  Vivint Camera  │
│      App        │   devices/<device_id>/msg     │   (Doorbell)    │
└────────┬────────┘                               └────────┬────────┘
         │                                                  │
         │  WebRTC Peer Connection                          │
         │  (STUN/TURN via v2.nuts.vivint.ai)              │
         │◄────────────────────────────────────────────────►│
         │                                                  │
         │  DataChannel (protobuf messages)                 │
         │  - TwoWayTalkStart                               │
         │  - TwoWayTalkEnd                                 │
         │  - Ping/Pong                                     │
         └──────────────────────────────────────────────────┘
```

## Authentication Flow

### Step 1: Get Vivint OAuth Token
Use the existing Vivint API OAuth flow to obtain an access token.

### Step 2: Exchange for Firebase Custom Token
```
POST https://exchange.run.vivint.ai?custom-token=true
Authorization: Bearer <vivint_oauth_token>

Response: {"custom-token": "<firebase_custom_token>"}
```

### Step 3: Firebase Authentication
```python
# Sign into Firebase with the custom token
firebase_auth.sign_in_with_custom_token(firebase_custom_token)
```

## STUN/TURN Servers

### Production Servers
| Type | URL | Port |
|------|-----|------|
| STUN | `stun:v2.nuts.vivint.ai` | 80 |
| TURN | `turn:v2.nuts.vivint.ai` | 80 |

### TURN Credentials
- **Username**: `coturn@vivint.com`
- **Password**: `VivCamProCoturnAuthAndCred{3779}`

### Test Servers (for development)
| Type | URL |
|------|-----|
| STUN | `stun:v2.test.nuts.vivint.ai:80` |
| TURN | `turn:v2.test.nuts.vivint.ai:80` |

Same credentials as production.

## Firebase Database Paths

### Signaling Path
```
devices/<camera_device_id>/msg
```
Or with system grouping:
```
groups/<system_id>/devices/<camera_device_id>/msg
```

### Message Direction
- **Client → Camera**: Write to camera's `/msg` path
- **Camera → Client**: Listen on client's `/msg` path

## Signaling Messages

### SdpMessage (for SDP offer/answer)
```json
{
  "type": "sdp",
  "sdpType": "offer",           // or "answer"
  "sdp": "<SDP string>",
  "transactionUuid": "<uuid>",
  "from": "<app_uuid>",
  "videoSource": {"type": "nosource"},  // or null
  "_timestamp": 1234567890123
}
```

### IceMessage (for ICE candidates)
```json
{
  "type": "ice",
  "candidate": "<ice_candidate_string>",
  "sdpMLineIndex": 0,
  "from": "<app_uuid>",
  "_timestamp": 1234567890123
}
```

## WebRTC Connection Flow

```
1. App authenticates with Firebase using custom token

2. App creates PeerConnection with STUN/TURN servers:
   - Creates DataChannel named "channel"
   - Creates SDP offer
   - Sets local description

3. App sends SDP offer to camera via Firebase:
   - Path: devices/<camera_id>/msg
   - Message: SdpMessage with sdpType="offer"

4. App listens for camera response on its own path:
   - Path: devices/<app_uuid>/msg

5. Camera sends SDP answer via Firebase

6. ICE candidates exchanged bidirectionally via Firebase

7. WebRTC connection established

8. DataChannel used for control messages
```

## DataChannel Protocol (Protobuf)

The DataChannel uses protobuf-encoded `DataChannelMessage` with these message types:

### Control Messages
| Field Number | Message Type | Description |
|--------------|--------------|-------------|
| 1 | Ping | Keep-alive ping |
| 2 | Pong | Keep-alive response |
| 9 | TwoWayTalkStart | Start two-way audio |
| 10 | TwoWayTalkStartResponse | Camera acknowledgment |
| 11 | TwoWayTalkEnd | Stop two-way audio |
| 12 | TwoWayTalkEndResponse | Camera acknowledgment |

### DVR/Playback Messages
| Field Number | Message Type | Description |
|--------------|--------------|-------------|
| 15 | PlayPauseSeek | DVR playback control |
| 16 | GetOldestTimestamp | Get earliest recording |
| 18 | GetClipRequest | Request video clip |
| 20 | GetMetadataRequest | Get camera metadata |

### Camera Settings
| Field Number | Message Type | Description |
|--------------|--------------|-------------|
| 5 | SetCameraSettings | Change settings |
| 7 | GetCameraSettings | Query settings |
| 25 | GetDptzConfigRequest | Get PTZ config |

## Two-Way Talk Implementation

### Starting Two-Way Audio
```python
# Build protobuf message
message = DataChannelMessage()
message.two_way_talk_start.CopyFrom(TwoWayTalkStart())

# Send over DataChannel as binary
data_channel.send(message.SerializeToString(), is_binary=True)
```

### Stopping Two-Way Audio
```python
message = DataChannelMessage()
message.two_way_talk_end.CopyFrom(TwoWayTalkEnd())
data_channel.send(message.SerializeToString(), is_binary=True)
```

### Audio Track Configuration
- After sending `TwoWayTalkStart`, enable local audio track on the PeerConnection
- The camera will then accept audio from the client's microphone
- Audio is transmitted via WebRTC's audio track (not DataChannel)

## Implementation Checklist

1. [x] Obtain Vivint OAuth token (existing API - use id_token)
2. [x] Exchange for Firebase custom token
3. [x] Exchange custom token for Firebase ID token (signInWithCustomToken)
4. [x] Connect to Firebase WebSocket (NOT REST API - REST is blocked by security rules)
5. [x] Authenticate WebSocket with Firebase ID token
6. [x] Subscribe to signaling paths
7. [x] Create PeerConnection with STUN/TURN config
8. [x] Create DataChannel named "channel"
9. [x] Generate SDP offer
10. [x] Wait for ICE gathering (TURN allocation)
11. [x] Send offer via Firebase WebSocket
12. [x] Extract and send ICE candidates separately via Firebase
13. [x] Listen for answer from camera
14. [x] Handle remote ICE candidates
15. [x] Establish WebRTC connection
16. [x] Send Ping, wait for Pong
17. [x] Send TwoWayTalkStart message
18. [x] Receive TwoWayTalkStartResponse confirmation
19. [x] Enable audio track (MediaPlayer with DirectShow on Windows)
20. [x] Send/receive audio (confirmed working with TwoWayTalkStartResponse)
21. [x] Send TwoWayTalkEnd to stop

## IMPORTANT Implementation Notes

### Firebase REST API Does NOT Work
Firebase security rules block REST API access. You MUST use Firebase WebSocket protocol:
- Connect to: `wss://<firebase-db-url>/.ws?v=5&ns=<namespace>`
- Auth with: `{"t":"d","d":{"r":1,"a":"auth","b":{"cred":"<id_token>"}}}`
- Write with: `{"t":"d","d":{"r":2,"a":"p","b":{"p":"/<path>","d":<data>}}}`
- Subscribe with: `{"t":"d","d":{"r":3,"a":"q","b":{"p":"/<path>","h":""}}}`

### Device UUID Mapping
The camera's device UUID is NOT the MAC address! Find it in `camera.data['uuid']`.

### ICE Candidate Exchange
aiortc does not emit browser-style "icecandidate" events. Extract candidates from
the local description SDP after ICE gathering and send them via Firebase:
```python
candidate_pattern = r'a=candidate:(.+)'
candidates = re.findall(candidate_pattern, local_desc.sdp)
for candidate in candidates:
    await signaler.send_ice_candidate(f"candidate:{candidate}", 0)
```

### Stale Data Handling
Firebase subscriptions return current data at the path. Filter out stale messages:
- Check if local description is set before processing answers
- Optionally validate transaction UUIDs

## Key Files in Decompiled Code

| File | Purpose |
|------|---------|
| `StandaloneSignalerImpl.java` | Main signaling implementation |
| `PeerConnectionClient.java` | WebRTC peer connection management |
| `StandaloneAuthorizerBaseImpl.java` | Firebase token exchange |
| `DataChannel.java` | Protobuf message definitions |
| `SdpMessage.java` | SDP message structure |
| `IceMessage.java` | ICE candidate message structure |
| `BuildConfig.java` | STUN/TURN server URLs and credentials |

## Notes

- The camera generates the SDP offer first in some scenarios
- Firebase paths may include system grouping for multi-panel setups
- All timestamps are Unix milliseconds
- The app generates a unique UUID for each session (`appUuid`/`from` field)
- DataChannel messages are binary protobuf, not JSON
