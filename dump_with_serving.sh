#!/bin/bash
# Script to run MOE dumping with background serving for DeepSeek-V3

# Configuration
MODEL="deepseek-ai/DeepSeek-V3"
DUMP_DIR="/workspace/moe_dumps"
MAX_DUMPS=5

echo "🚀 Starting serving endpoint with dumping initially DISABLED..."
echo "Model: $MODEL"
echo "Dump directory: $DUMP_DIR"
echo "Max dumps: $MAX_DUMPS"

# Create dump directory if it doesn't exist
mkdir -p "$DUMP_DIR"

# Start with dumping disabled
export DUMP_MOE_INPUTS=0
export MOE_DUMP_DIR="$DUMP_DIR"
export MOE_MAX_DUMPS=$MAX_DUMPS

# Clean up any existing control file and old dumps
rm -f /tmp/moe_dump_enabled
echo "Cleaned up control file"

# Optional: Clear old dumps
# rm -rf "$DUMP_DIR"/request_*

# Start the server in background, capture output
SERVER_LOG="/tmp/sglang_server_$$.log"
echo ""
echo "📦 Starting server (dumping disabled during warmup)..."
echo "Server log: $SERVER_LOG"

python3 -m sglang.launch_server \
    --model "$MODEL" \
    --trust-remote-code \
    --tp-size 8 \
    --moe-runner-backend flashinfer_trtllm \
    --quantization fp8 2>&1 | tee "$SERVER_LOG" &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready by checking for the ready message
echo ""
echo "⏳ Waiting for server to be ready (checking for ready message)..."
echo "This may take ~3.5 minutes..."

MAX_WAIT=300  # 5 minutes timeout
WAIT_INTERVAL=5
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    # Check if server is still running
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "❌ Server process died! Check the logs:"
        tail -50 "$SERVER_LOG"
        exit 1
    fi
    
    # Check for the ready message
    if grep -q "The server is fired up and ready to roll!" "$SERVER_LOG" 2>/dev/null; then
        echo "✅ Server is ready!"
        break
    fi
    
    # Show progress
    if [ $((ELAPSED % 30)) -eq 0 ] && [ $ELAPSED -gt 0 ]; then
        echo "  Still waiting... ($ELAPSED seconds elapsed)"
    fi
    
    sleep $WAIT_INTERVAL
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
done

# Check if we timed out
if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "❌ Timeout waiting for server to be ready!"
    echo "Last 50 lines of server log:"
    tail -50 "$SERVER_LOG"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

echo "Server startup took approximately $ELAPSED seconds"

# Enable dumping using the CONTROL FILE (not environment variable!)
echo ""
echo "✅ Enabling dumping via control file..."
touch /tmp/moe_dump_enabled
echo "Control file created: /tmp/moe_dump_enabled"
ls -la /tmp/moe_dump_enabled

# Now run your benchmark - dumps will be created!
echo ""
echo "📊 Running benchmark (dumps will be created)..."
echo "Command: python3 -m sglang.bench_serving --backend sglang --model $MODEL --num-prompts 80 --sharegpt-output-len 100 --max-concurrency 1 --warmup-requests 0"
echo ""

python3 -m sglang.bench_serving \
    --backend sglang \
    --model "$MODEL" \
    --num-prompts 80 \
    --sharegpt-output-len 100 \
    --max-concurrency 1 \
    --warmup-requests 0

echo ""
echo "📊 Benchmark completed!"

# Disable dumping
echo ""
echo "🛑 Disabling dumping..."
rm -f /tmp/moe_dump_enabled

# Show dump statistics
echo ""
echo "📁 Dump statistics:"
if [ -d "$DUMP_DIR" ]; then
    NUM_DUMPS=$(ls -1 "$DUMP_DIR"/request_* 2>/dev/null | wc -l)
    echo "Total dumps created: $NUM_DUMPS"
    if [ $NUM_DUMPS -gt 0 ]; then
        echo "Dump locations:"
        ls -la "$DUMP_DIR" | head -20
        echo ""
        echo "Sample dump contents:"
        ls -la "$DUMP_DIR/request_000/" 2>/dev/null || echo "No request_000 found"
    fi
else
    echo "No dumps directory found!"
fi

# Kill the server
echo ""
echo "🛑 Stopping server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null

echo ""
echo "✅ Complete! Dumps are in: $DUMP_DIR"

# Cleanup
rm -f "$SERVER_LOG"
