#!/bin/bash
set -e

ENV=${1:?Usage: $0 {dev|stg|pro}}
export APP_ENV=$ENV

case $ENV in
    dev) PORT=37709; WORKERS=2 ;;
    stg) PORT=80;    WORKERS=4 ;;
    pro) PORT=80;    WORKERS=6 ;;
    *)   echo "Unknown env: $ENV"; exit 1 ;;
esac

echo "[$ENV] Stopping existing processes..."
pkill -f "gunicorn.*server:app" || true
echo "stop done"

echo "sleep 10 seconds..."
sleep 10

rm -f ./log/deploy_test.log ./log/gunicorn_log.log

nohup gunicorn -w $WORKERS -t 1000 -b 0.0.0.0:$PORT server:app >> ./log/gunicorn_log.log 2>&1 &
echo "restart done"

sleep 10
echo "[ailoan cvmodels] deploy check..."
nohup python ./unit_test/test_localhost.py >> ./log/deploy_test.log 2>&1 &

sleep 2
curl -s http://127.0.0.1:$PORT/version/ >> ./log/deploy_test.log
echo "done"
