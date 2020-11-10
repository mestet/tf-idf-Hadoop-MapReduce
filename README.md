TF-IDF Hadoop MapReduce
=======================

Реализация TF-IDF алгоритма.
<br>
Forked from https://github.com/ziky90/tf-idf-Hadoop-MapReduce 

<hr>
Input to the mapreduce program should be <b>raw documents</b> where each of them is <b>represented as a line</b>

text text text tex<br>
txt text word<br>
.<br>
.<br>
.<br>
xword yword<br>

<hr>
Output of this program will be the final TF-IDF matrix:

doc1  car:0.9 plane:1.6 computer:2.3<br>
doc2  elephant:0.2 hadoop:1.1<br>
.<br>
.<br>
.<br>
docn  mahout:0.8 storm:1.6<br>

<hr>
<b>Usage:</b>

<b>1)</b> You need to create jar:
use command: <i>mvn clean package</i>

<b>2)</b> Copy the jar to the machine where you have your Hadoop installed

<b>3)</b> Run program:
<i>hadoop jar [name of the jar] com.zikesjan.bigdata.TfIdfMain [input path] [output path]</i>

