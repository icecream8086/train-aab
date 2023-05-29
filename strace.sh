rm -rf trace.log
strace -c -f python server.py 2> trace.log
