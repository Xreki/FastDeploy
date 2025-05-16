
export HOST=${HOST:-"10.11.155.41:8771"}

#获取job1全部实例：
#curl -X GET http://$HOST/api/v1/job/1
#删除job1的实例xx：
#curl -X DELETE http://$HOST/api/v1/job/1/instance/xx