# BUG记录

在new 和delete的时候出现segment fault，原因是在某一次出现的访问越界问题

顺序不对，encode中第10，11访问到的数据，在decode中第6，7个被访问到

9->6
10->9
11->7

3
