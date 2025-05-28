package main

import (
    "context"
    "encoding/json"
    "log"
    "google.golang.org/grpc"
)

func main() {
    conn, err := grpc.Dial("localhost:50051", grpc.WithInsecure())
    if err != nil {
        log.Fatal(err)
    }
    defer conn.Close()
    send := conn.Invoke
    ctx := context.Background()
    if err := send(ctx, "/bus.Bus/Send", []byte("proto_schema=1"), nil); err != nil {
        log.Fatal(err)
    }
    msg := map[string]any{
        "sender": "cli",
        "recipient": "orch",
        "payload": map[string]any{"hello": "world"},
        "ts": 0.0,
    }
    data, _ := json.Marshal(msg)
    if err := send(ctx, "/bus.Bus/Send", data, nil); err != nil {
        log.Fatal(err)
    }
}
