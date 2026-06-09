# 停止现有进程
echo "stop ..."
ps -efww | grep 37709 | awk '{print $2}' | xargs kill -9
rm ./log/deploy_test.log
rm ./log/gunicorn_log.log
echo "stop done ..."

# 休眠10秒以确保下次重启稳定
echo "sleep 10 seconds ..."
sleep 10

nohup gunicorn -w 2 -t 1000 -b 0.0.0.0:37709 server:app >> ./log/gunicorn_log.log &
echo "restart done..."

sleep 10
echo "[ailoan cvmodels] deploy check..."
nohup python ./unit_test/test_localhost.py >> ./log/deploy_test.log &

sleep 2
echo curl http://127.0.0.1:37709/version/ >> ./log/deploy_test.log
echo "done"