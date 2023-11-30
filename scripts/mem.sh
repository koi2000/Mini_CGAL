#!/bin/bash

#用法：输入“./monitor.sh compiz 1”或者“./monitor.sh compiz 1 logon”
#其中：compiz表示要监控的进程名称，1表示要刷新的时间间隔，logon表示将监测数据同步输出到记录文件中，该参数为可选参数，如果不写则表示不输出到记录文件

PROG_NAME=$1
SLEP_TIME=$2
LOG_ENABL=$3
SDIR_PATH="monitor_log"
FILE_NAME=${PROG_NAME}_$(date "+%Y%m%d_%H%M%S").log

if [ "$LOG_ENABL" = "logon" ];then
	if [ ! -d ${SDIR_PATH} ];then
		mkdir ${SDIR_PATH}
	fi
	cd ${SDIR_PATH}
	echo "log is enabled. log_file \"${FILE_NAME}\""
fi

pre_pro_cpu=""
pre_sys_cpu=""

while [ 1 ]
do
	PID=$(ps -A | grep $PROG_NAME | grep -v 'grep' | awk '{print $1;}')
	time=$(date "+%Y/%m/%d %H:%M:%S")
	
	if [ "$PID" != "" ];then
		pro_tu_time=$(cat /proc/$PID/stat | awk -F " " '{print $14}')
		pro_ts_time=$(cat /proc/$PID/stat | awk -F " " '{print $15}')
		pro_cu_time=$(cat /proc/$PID/stat | awk -F " " '{print $16}')
		pro_cs_time=$(cat /proc/$PID/stat | awk -F " " '{print $17}')
		cur_prc_cpu=$((10#${pro_tu_time}+10#${pro_ts_time}+10#${pro_cu_time}+10#${pro_cs_time}))
		
		sys_us_time=$(cat /proc/stat | awk -F " " '{print $2}' | sed -n '1p')
		sys_ni_time=$(cat /proc/stat | awk -F " " '{print $3}' | sed -n '1p')
		sys_sy_time=$(cat /proc/stat | awk -F " " '{print $4}' | sed -n '1p')
		sys_id_time=$(cat /proc/stat | awk -F " " '{print $5}' | sed -n '1p')
		cur_sys_cpu=$((10#${sys_us_time}+10#${sys_ni_time}+10#${sys_sy_time}+10#${sys_id_time}))
		
		if [ "${pre_pro_cpu}" = "" ];then
			dif_pre_cpu=0
			dif_cur_cpu=1
		else
			dif_pre_cpu=$(((10#${cur_prc_cpu}-10#${pre_pro_cpu})))
			dif_cur_cpu=$(((10#${cur_sys_cpu}-10#${pre_sys_cpu})))			
		fi
		
		cpu_value=`awk -v x=${dif_pre_cpu} -v y=${dif_cur_cpu} 'BEGIN{printf "%.2f\n",100*x/y}'`
		pre_pro_cpu=${cur_prc_cpu}
		pre_sys_cpu=${cur_sys_cpu}
		
		mem_value=$(cat /proc/$PID/status | grep RSS | awk -F " " '{print $2}')
		disp="[${time}] cpu ${cpu_value}%, mem ${mem_value} KB"
	else
		disp="[${time}] proc \"${PROG_NAME}\" not found!!!"
	fi
	
	echo "${disp}"
	
	if [ "$LOG_ENABL" = "logon" ];then
		echo ${disp} >> "$FILE_NAME"
	fi
	
	sleep ${SLEP_TIME}
done
